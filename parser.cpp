#include "parser.h"
#include <string_view>
#include <iostream>
#include <ostream>

void print_error(unsigned int line_num, std::string_view message)
{
    std::cerr << "Line " << line_num << ": "
              << message << "\n";
}

void print_error(std::string_view message)
{
    std::cerr << message << "\n";
}

Parser::Parser(TokenIterator input_begin,
               TokenIterator input_end)
    : m_input_begin(input_begin), m_input_end(input_end)
{
    // Built-in functions/names (eventually, the standard library)
    m_names_table["print"] = NameType::Funct;
    m_names_table["max"] = NameType::Funct;
}

std::unique_ptr<Expression> Parser::in_literal()
{
    const TokenIterator current = token++;

    switch(current->type) {
    case TokenType::String_Literal:
        return std::unique_ptr<Expression>(new Literal(current->text));
    case TokenType::Char_Literal:
        return std::unique_ptr<Expression>(new Literal(current->text[0]));
    case TokenType::Int_Literal:
        return std::unique_ptr<Expression>(new Literal(std::stoi(current->text)));
    case TokenType::Float_Literal:
        return std::unique_ptr<Expression>(new Literal(std::stof(current->text)));
    default:
        print_error(current->line_num, "Expected a literal here, but instead found token:");
        std::cerr << *current;
        exit(1);
    }
}


std::unique_ptr<Expression> Parser::in_multiply_divide()
{
    std::unique_ptr<Expression> left_side = in_basic_expression();

    while(token->type == TokenType::Op_Mult
          || token->type == TokenType::Op_Div) {
        TokenType op = token->type;
        ++token;
        std::unique_ptr<Expression> right_side = in_basic_expression();
        left_side = std::make_unique<CompositeExpression>(std::move(left_side), op,
                                                          std::move(right_side));
    }

    return left_side;
}

std::unique_ptr<Expression> Parser::in_add_subtract()
{
    std::unique_ptr<Expression> left_side = in_multiply_divide();

    while(token->type == TokenType::Op_Plus
          || token->type == TokenType::Op_Minus) {
        TokenType op = token->type;
        ++token;
        std::unique_ptr<Expression> right_side = in_multiply_divide();
        left_side = std::make_unique<CompositeExpression>(std::move(left_side), op,
                                                          std::move(right_side));
    }

    return left_side;
}

std::unique_ptr<Expression> Parser::in_basic_expression()
{
    // TODO: add support for LValues, too
    switch(token->type) {
    case TokenType::Name:
        return in_function_call();
    case TokenType::Open_Parentheses:
        return in_parentheses();
    default:
        return in_literal();
    }
}

std::unique_ptr<Expression> Parser::in_composite_expression()
{
    return in_add_subtract();
}

std::unique_ptr<FunctionCall> Parser::in_function_call()
{
    auto new_function_call = std::make_unique<FunctionCall>();

    // First, assign the function call's name, if it's valid
    if(token->type == TokenType::Name && m_names_table.count(token->text) > 0) {
        if(m_names_table[token->text] == NameType::Funct) {
            new_function_call->name = token->text;
        } else {
            print_error(token->line_num, "Name `" + token->text + "` isn't a "
                        + "function name, but is being used as one");
            exit(1);
        }
    } else if(token->type != TokenType::Name) {
        print_error(token->line_num, "Function call doesn't begin with a valid name");
        exit(1);
    } else {
        print_error(token->line_num, "Undefined function: `" + token->text + "`");
        exit(1);
    }

    ++token;
    if(token->type != TokenType::Open_Parentheses) {
        print_error(token->line_num, "Expected '(' token before function call argument list");
        exit(1);
    }

    // Next, parse the arguments, each of which should be some sort of expression
    ++token;
    while(token != m_input_end) {
        switch(token->type) {
        case TokenType::Op_Comma:
            // The comma after an argument expression; continue
            ++token;
            break;
        case TokenType::Closed_Parentheses:
            ++token;
            return new_function_call;
        default:
            new_function_call->arguments.push_back(in_expression());
        }
    }
    print_error(token->line_num, "Function call definition ended early");
    exit(1);
}

std::unique_ptr<Expression> Parser::in_parentheses()
{
    if(token->type != TokenType::Open_Parentheses) {
        print_error(token->line_num, "Expected open parentheses for an expression group, but instead found token:");
        std::cerr << *token;
        exit(1);
    }

    ++token;
    std::unique_ptr<Expression> result = in_expression();
    if(token->type != TokenType::Closed_Parentheses) {
        print_error(token->line_num, "Expected closing parentheses for an expression group, but instead found token:");
        std::cerr << *token;
        exit(1);
    }

    ++token;
    return result;
}

std::unique_ptr<Expression> Parser::in_expression()
{
    const TokenType next_type = std::next(token)->type;;

    if(token->type == TokenType::Name && next_type == TokenType::Open_Parentheses) {
        return in_function_call();
    } else if(next_type == TokenType::Op_Plus || next_type == TokenType::Op_Minus
              || next_type == TokenType::Op_Div || next_type == TokenType::Op_Mult) {
        return in_composite_expression();
    } else {
        return in_basic_expression();
    }
}


std::unique_ptr<Assignment> Parser::in_assignment()
{
    if(token->type != TokenType::Keyword_Var) {
        print_error(token->line_num, "Expected keyword `var`, but instead got token:");
        std::cerr << *token << '\n';
        exit(1);
    }

    auto new_statement = std::make_unique<Assignment>();
    ++token;
    if(token->type != TokenType::Name) {
        print_error(token->line_num,
                    "Expected the name of an lvalue, but instead got token:");
        std::cerr << *token << '\n';
        exit(1);
    } else if(const auto match = m_names_table.find(token->text);
              match != m_names_table.end()) {
        print_error(token->line_num, "Name `" + token->text + "` is already in use");
        exit(1);
    }

    // Add this new lvalue to list of tracked names
    m_names_table[token->text] = NameType::LValue;
    auto new_lvalue = std::make_unique<Variable>();
    new_lvalue->name = token->text;
    m_lvalues.push_back(std::move(new_lvalue));
    new_statement->target = m_lvalues.back().get();

    ++token;
    if(token->type != TokenType::Type_Indicator) {
        print_error(token->line_num, "Expected `:` before typename, but instead got token:");
        std::cerr << *token << '\n';
        exit(1);
    }

    ++token;
    if(token->type != TokenType::Name) {
        print_error(token->line_num, "Expected typename, but instead got token:");
        std::cerr << *token << '\n';
        exit(1);
    } else if(const auto match = m_names_table.find(token->text);
              match == m_names_table.end()) {
        print_error(token->line_num, "Unknown typename: `" + token->text + "`");
        exit(1);
    }

    ++token;
    if(token->type != TokenType::Op_Assign) {
        print_error(token->line_num, "Expected assignment operator, but instead got token:");
        std::cerr << *token << '\n';
        exit(1);
    }

    ++token;
    // Set the expression after the assignment operator to be the subexpression
    // in the statement
    while(token != m_input_end) {
        if(token->type == TokenType::End_Statement) {
            // End of this statement; go back to prior state
            ++token;
            return new_statement;
        } else {
            new_statement->expression = in_expression();
        }
    }
    print_error(token->line_num, "Assignment statement ended early");
    exit(1);
}

std::unique_ptr<Statement> Parser::in_statement()
{
    auto new_statement = std::make_unique<Statement>();

    while(token != m_input_end) {
        // TODO: handle statements not inside a funct
        if(token->type == TokenType::End_Statement) {
            // End of this statement; go back to prior state
            ++token;
            return new_statement;
        } else if(token->type == TokenType::Keyword_Var) {
            // TODO: reorganize Statement class to have more utility, since it
            // will only hold a single expression
            // TODO: add support for constant lvalues
            return in_assignment();
        } else if(std::next(token)->type != TokenType::End_Statement) {
            // TODO: add support for variables with statements, too
            new_statement->expression = in_expression();
        } else {
            print_error(token->line_num, "Unexpected token:");
            std::cerr << *token << '\n';
            exit(1);
        }
    }
    print_error(token->line_num, "Statement ended early");
    exit(1);
}

void Parser::in_function_definition()
{
    Function new_funct;

    // First, set the function's name, making sure it isn't being used for
    // anything else
    ++token;
    if(token->type == TokenType::Name && m_names_table.count(token->text) == 0) {
        new_funct.name = token->text;
        m_names_table[new_funct.name] = NameType::Funct;
    } else if(token->type != TokenType::Name) {
        print_error(token->line_num, "Expected name to follow `funct` keyword");
        exit(1);
    } else {
        print_error(token->line_num, "Name `" + token->text + "` is"
                    + " already in use");
        exit(1);
    }

    ++token;
    if(token->type != TokenType::Open_Parentheses) {
        print_error(token->line_num,
                    "Expected `(` to follow name of function in definition");
        exit(1);
    }

    // TODO: add support for type annotations next to param names
    //      and commas after each name/type
    // Next, parse the parameters
    ++token;
    while(token != m_input_end) {
        if(token->type == TokenType::Name) {
            new_funct.parameters.push_back(token->text);
        } else if(token->type == TokenType::Closed_Parentheses) {
            // Next, look for `is` keyword (TODO: open a new scope here)
            ++token;
            if(token->type != TokenType::Keyword_Is) {
                print_error(token->line_num,
                            "Expected keyword `is` to follow parameters of the function");
                exit(1);
            }
            break;
        } else {
            print_error(token->line_num, "Expected parameter name");
            exit(1);
        }

        ++token;
    }

    // Finally, parse the body of the function
    ++token;
    while(token != m_input_end) {
        if(token->type != TokenType::Keyword_End) {
            // Found a statement, parse it
            new_funct.statements.push_back(in_statement());
        } else {
            // End of function (TODO: close scope here)
            ++token; // Take the name associated with `end` (e.g. `main` in `end main`)
            if(token->type != TokenType::Name && token->text != new_funct.name) {
                print_error(token->line_num,
                            "No matching `end " + new_funct.name + "` for"
                            + " `funct` " + new_funct.name);
                exit(1);
            }
            m_names_table[new_funct.name] = NameType::Funct;
            m_functions.push_back(std::move(new_funct));
            ++token;
            return;
        }
    }
}

void Parser::run()
{
    token = m_input_begin;
    while(token != m_input_end) {
        switch(token->type) {
        case TokenType::Keyword_Funct:
            // Found start of a function definition
            in_function_definition();
            break;
        default:
            print_error(token->line_num, "Unexpected token:");
            std::cerr << *token << '\n';
            exit(1);
        }
    }
}


CompositeExpression::CompositeExpression(std::unique_ptr<Expression>&& l, TokenType oper,
                                         std::unique_ptr<Expression>&& r)
    : left(std::move(l)), op(oper), right(std::move(r))
{}

void CompositeExpression::print(std::ostream& output) const
{
    output << '(';
    left->print(output);
    output << ' ' << op << ' ';
    right->print(output);
    output << ")\n";
}

void FunctionCall::print(std::ostream& output) const
{
    output << name << '(';
    for(const auto& argument : arguments) {
        argument->print(output);
        output << ", ";
    }
    output << ')';
}

void Statement::print(std::ostream& output) const
{
    output << "Statement:\n";
    expression->print(output);
    output << '\n';
}

void Variable::print(std::ostream& output) const
{
    output << "Variable: " << name << '\n';
}

std::ostream& operator<<(std::ostream& output, const Function& function)
{
    output << "Function: " << function.name << '\n';
    output << "Parameters:\n";
    for(const auto& param : function.parameters) {
        output << param << ' ';
    }
    output << "\nStatements:\n";

    for(const auto& statement : function.statements) {
        statement->print(output);
    }
    return output;
}

std::ostream& operator<<(std::ostream& output, const Parser& parser)
{
    for(const auto& function_definition : parser.m_functions) {
        output << function_definition << '\n';
    }
    return output;
}
