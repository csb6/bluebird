#include "parser.h"
#include <string_view>
#include <iostream>

void print_error(unsigned int line_num, std::string_view message)
{
    std::cerr << "Line " << line_num << ": "
              << message << "\n";
}

void print_error(std::string_view message)
{
    std::cerr << message << "\n";
}

void print_error_expected(std::string_view expected, Token actual)
{
    std::cerr << "Line " << actual.line_num << ": "
              << "Expected " << expected << ", but instead found token:\n";
    std::cerr << actual;
}

Parser::Parser(TokenIterator input_begin,
               TokenIterator input_end)
    : m_input_begin(input_begin), m_input_end(input_end)
{
    // Built-in functions/names (eventually, the standard library)
    m_names_table["print"] = NameType::Funct;
    m_names_table["max"] = NameType::Funct;
}

Expression* Parser::parse_expression(TokenType right_token)
{
    Expression* left_side = in_expression();
    const Precedence right_precedence = precedence_of(right_token);
    Precedence curr_precedence = precedence_of(token->type);
    while(right_precedence < curr_precedence && token->type != TokenType::End_Statement) {
        TokenType op = token->type;
        if(!is_operator(op)) {
            print_error_expected("operator", *token);
            exit(1);
        }
        ++token;
        Expression* right_side = parse_expression(op);
        left_side = new CompositeExpression(left_side, op, right_side);
        curr_precedence = precedence_of(token->type);
    }
    return left_side;
}

Expression* Parser::in_literal()
{
    const TokenIterator current = token++;

    switch(current->type) {
    case TokenType::String_Literal:
        return new Literal(current->text);
    case TokenType::Char_Literal:
        return new Literal(current->text[0]);
    case TokenType::Int_Literal:
        return new Literal(std::stoi(current->text));
    case TokenType::Float_Literal:
        return new Literal(std::stof(current->text));
    default:
        print_error_expected("literal", *current);
        exit(1);
    }
}

Expression* Parser::in_lvalue_expression()
{
    const TokenIterator current = token++;

    if(current->type != TokenType::Name) {
        print_error_expected("variable/constant name", *token);
        exit(1);
    }

    const auto match = m_names_table.find(current->text);
    if(match == m_names_table.end()) {
        print_error("Unknown lvalue `" + current->text + "`");
        exit(1);
    } else if(match->second != NameType::LValue) {
        print_error("Expected name of variable/constant, but `"
                    + match->first + "` is already being used as a name");
        exit(1);
    }

    auto new_lvalue_expr = new LValueExpression();
    new_lvalue_expr->name = current->text;
    return new_lvalue_expr;
}

Expression* Parser::in_expression()
{
    switch(token->type) {
    case TokenType::Name:
        if(std::next(token)->type == TokenType::Open_Parentheses) {
            return in_function_call();
        } else {
            return in_lvalue_expression();
        }
    case TokenType::Open_Parentheses:
        return in_parentheses();
    default:
        return in_literal();
    }
}

FunctionCall* Parser::in_function_call()
{
    auto* new_function_call = new FunctionCall();

    if(token->type != TokenType::Name) {
        print_error_expected("function name", *token);
        exit(1);
    }

    // First, assign the function call's name
    const auto match = m_names_table.find(token->text);
    if(match == m_names_table.end()) {
        // If the function hasn't been declared yet, add it provisionally to name table
        // to be filled in (hopefully) later
        m_names_table[token->text] = NameType::DeclaredFunct;
    } else if(match->second != NameType::Funct && match->second != NameType::DeclaredFunct) {
        print_error(token->line_num, "Expected `" + token->text
                    + "` to be a function name, but it is defined as another kind of name");
        exit(1);
    }

    new_function_call->name = token->text;

    ++token;
    if(token->type != TokenType::Open_Parentheses) {
        print_error_expected("'(' token before function call argument list", *token);
        exit(1);
    }

    // Next, parse the arguments, each of which should be some sort of expression
    ++token;
    while(token != m_input_end) {
        switch(token->type) {
        case TokenType::Comma:
            // The comma after an argument expression; continue
            ++token;
            break;
        case TokenType::Closed_Parentheses:
            ++token;
            return new_function_call;
        default:
            new_function_call->arguments.emplace_back(parse_expression(TokenType::Comma));
        }
    }
    print_error(token->line_num, "Function call definition ended early");
    exit(1);
}

Expression* Parser::in_parentheses()
{
    if(token->type != TokenType::Open_Parentheses) {
        print_error_expected("opening parentheses for an expression group", *token);
        exit(1);
    }

    ++token;
    Expression* result = parse_expression(TokenType::Closed_Parentheses);
    if(token->type != TokenType::Closed_Parentheses) {
        print_error_expected("closing parentheses for an expression group", *token);
        exit(1);
    }

    ++token;
    return result;
}

Magnum::Pointer<LValue> Parser::in_lvalue_declaration()
{
    if(token->type != TokenType::Name) {
        print_error_expected("the name of an lvalue", *token);
        exit(1);
    }
    Magnum::Pointer<LValue> new_lvalue;
    new_lvalue = Magnum::pointer<LValue>();

    new_lvalue->name = token->text;

    ++token;
    if(token->type != TokenType::Type_Indicator) {
        print_error_expected("`:` before typename", *token);
        exit(1);
    }

    ++token;
    if(token->type == TokenType::Keyword_Const) {
        // `constant` keyword marks kind of lvalue
        new_lvalue->is_mutable = false;
        ++token;
    }
    if(token->type != TokenType::Name) {
        print_error_expected("typename", *token);
        exit(1);
    }

    const auto match = m_names_table.find(token->text);
    if(match == m_names_table.end()) {
        // If the type hasn't been declared yet, add it provisionally to name table
        // to be filled in (hopefully) later
        m_names_table[token->text] = NameType::DeclaredType;
    } else if(match->second != NameType::Type && match->second != NameType::DeclaredType) {
        print_error(token->line_num, "Expected `" + token->text
                    + "` to be a typename, but it is defined as another kind of name");
        exit(1);
    }

    new_lvalue->type = token->text;
    return new_lvalue;
}

Magnum::Pointer<Initialization> Parser::in_initialization()
{
    if(token->type != TokenType::Keyword_Let) {
        print_error_expected("keyword `let`", *token);
        exit(1);
    }

    auto new_statement = Magnum::pointer<Initialization>();

    // Add this new lvalue to list of tracked names
    {
        ++token;
        Magnum::Pointer<LValue> new_lvalue = in_lvalue_declaration();
        if(const auto match = m_names_table.find(new_lvalue->name);
           match != m_names_table.end()) {
            print_error(token->line_num, "Name `" + token->text + "` is already in use");
            exit(1);
        }
        m_names_table[new_lvalue->name] = NameType::LValue;
        m_lvalues.push_back(std::move(new_lvalue));
        new_statement->target = m_lvalues.back().get();
    }

    ++token;
    if(token->type == TokenType::End_Statement) {
        // Declaration is valid, just no initial value was set
        ++token;
        return new_statement;
    } else if(token->type != TokenType::Op_Assign) {
        print_error_expected("assignment operator or ';'", *token);
        exit(1);
    }

    ++token;
    // Set the expression after the assignment operator to be the subexpression
    // in the statement
    new_statement->expression = Magnum::pointer<Expression>(parse_expression());

    ++token;
    return new_statement;
}

Magnum::Pointer<Statement> Parser::in_statement()
{
    switch(token->type) {
    case TokenType::Keyword_If:
        return in_if_block();
    case TokenType::Keyword_Let:
        return in_initialization();
    default:
        return in_basic_statement();
    };
}

// TODO: fix bug (remove semicolon from parser-test:8, run to see no error)
Magnum::Pointer<BasicStatement> Parser::in_basic_statement()
{
    auto new_statement = Magnum::pointer<BasicStatement>();
    new_statement->expression = Magnum::pointer<Expression>(parse_expression());

    if(token->type != TokenType::End_Statement) {
        print_error_expected("end of statement (a.k.a. `;`)", *token);
        exit(1);
    }

    ++token;
    return new_statement;
}

Magnum::Pointer<IfBlock> Parser::in_if_block()
{
    if(token->type != TokenType::Keyword_If) {
        print_error_expected("keyword if", *token);
        exit(1);
    }

    ++token;
    if(token->type != TokenType::Open_Parentheses) {
        print_error_expected("`(`", *token);
        exit(1);
    }

    ++token;
    auto new_block = Magnum::pointer<IfBlock>();
    
    // First, parse the if-condition
    new_block->condition = Magnum::pointer<Expression>(
        parse_expression(TokenType::Closed_Parentheses));

    if(token->type != TokenType::Closed_Parentheses) {
        print_error_expected("`)` to close `if` condition", *token);
        exit(1);
    }

    ++token;
    if(token->type != TokenType::Keyword_Do) {
        print_error_expected("keyword `do`", *token);
        exit(1);
    }

    // Next, parse the statements inside the if-block
    ++token;
    while(token != m_input_end) {
        if(token->type != TokenType::Keyword_End) {
            // Found a statement, parse it
            new_block->statements.push_back(in_statement());
        } else {
            // End of block (TODO: close scope here)
            ++token; // Take the `if` associated with `end`
            if(token->type != TokenType::Keyword_If) {
                print_error(token->line_num,
                            "No closing `end if` for if-block");
                exit(1);
            }

            ++token;
            if(token->type != TokenType::End_Statement) {
                print_error_expected("end of statement (a.k.a. `;`)", *token);
                exit(1);
            }

            ++token;
            break;
        }
    }

    return new_block;
}

void Parser::in_function_definition()
{
    Function new_funct;

    // First, set the function's name, making sure it isn't being used for
    // anything else
    ++token;
    if(token->type != TokenType::Name) {
        print_error_expected("name to follow `function keyword`", *token);
        exit(1);
    }

    const auto match = m_names_table.find(token->text);
    if(match == m_names_table.end() || match->second == NameType::DeclaredFunct) {
        new_funct.name = token->text;
        m_names_table[new_funct.name] = NameType::Funct;
    } else {
        print_error(token->line_num, "Name `" + token->text + "` is"
                    + " already in use");
        exit(1);
    }

    ++token;
    if(token->type != TokenType::Open_Parentheses) {
        print_error_expected("`(` to follow name of function in definition", *token);
        exit(1);
    }

    // Next, parse the parameters
    ++token;
    while(token != m_input_end) {
        if(token->type == TokenType::Name) {
            // Add a new parameter declaration
            new_funct.parameters.push_back(in_lvalue_declaration());

            ++token;
            // Check for a comma separator after typename
            if(token->type == TokenType::Closed_Parentheses) {
                // Put back so next iteration can handle end of parameter list
                --token;
            } else if(token->type != TokenType::Comma) {
                print_error_expected("comma to follow the parameter", *token);
                exit(1);
            }
        } else if(token->type == TokenType::Closed_Parentheses) {
            // Next, look for `is` keyword (TODO: open a new scope here)
            ++token;
            if(token->type != TokenType::Keyword_Is) {
                print_error_expected("keyword `is` to follow parameters of the function",
                                     *token);
                exit(1);
            }
            break;
        } else {
            print_error_expected("parameter name", *token);
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
                            + " `function` " + new_funct.name);
                exit(1);
            }

            ++token;
            if(token->type != TokenType::End_Statement) {
                print_error_expected("end of statement (a.k.a. `;`)", *token);
                exit(1);
            }

            m_names_table[new_funct.name] = NameType::Funct;
            m_functions.push_back(std::move(new_funct));
            ++token;
            return;
        }
    }
}

void Parser::in_type_definition()
{
    if(token->type != TokenType::Keyword_Type) {
        print_error_expected("keyword `type`", *token);
        exit(1);
    }

    ++token;
    if(token->type != TokenType::Name) {
        print_error_expected("typename", *token);
        exit(1);
    } else if(const auto match = m_names_table.find(token->text);
              match != m_names_table.end() && match->second != NameType::DeclaredType) {
        print_error(token->line_num, "Name: " + token->text + " already in use");
        exit(1);
    }

    m_names_table[token->text] = NameType::Type;
    m_types.push_back({token->text});

    // TODO: handle ranges/arrays/record types here

    ++token;
    if(token->type != TokenType::End_Statement) {
        print_error_expected("end of statement (a.k.a. `;`)", *token);
        exit(1);
    }
    ++token;
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
        case TokenType::Keyword_Type:
            in_type_definition();
            break;
        default:
            print_error(token->line_num, "Unexpected token:");
            std::cerr << *token << '\n';
            exit(1);
        }
    }
}

std::ostream& operator<<(std::ostream& output, const Parser& parser)
{
    for(const auto& function_definition : parser.m_functions) {
        output << function_definition << '\n';
    }
    return output;
}
