#include "parser.h"
#include <string_view>
#include <iostream>

void print_error(unsigned int line_num, std::string_view message)
{
    std::cerr << "Line " << line_num << ": "
              << message << "\n";
}

Parser::Parser(TokenIterator input_begin,
               TokenIterator input_end)
    : m_input_begin(input_begin), m_input_end(input_end)
{
    // Built-in functions/names (eventually, the standard library)
    m_names_table["print"] = NameType::Funct;
}

FunctionCall Parser::in_function_call()
{
    FunctionCall new_function_call;

    // First, assign the function call's name, if it's valid
    if(token->type == TokenType::Name && m_names_table.count(token->text) > 0) {
        if(m_names_table[token->text] == NameType::Funct) {
            new_function_call.name = token->text;
        } else {
            print_error(token->line_num, "Name `" + token->text + "` isn't a "
                        + "function name, but is being used as a function call");
            exit(1);
        }
    } else if(token->type != TokenType::Name) {
        print_error(token->line_num, "Function call doesn't begin with a valid name");
        exit(1);
    } else {
        print_error(token->line_num, "Undefined function: `" + token->text + "`");
        exit(1);
    }

    // Next, parse the arguments
    ++token;
    while(token != m_input_end) {
        if(token->type != TokenType::Closed_Parentheses) {
            new_function_call.arguments.push_back(token->text);
        } else {
            // End of this expression
            return new_function_call;
        }

        ++token;
    }
    print_error(token->line_num, "Function call definition ended early");
    exit(1);
}

Statement Parser::in_statement()
{
    Statement new_statement;

    while(token != m_input_end) {
        // TODO: handle statements not inside a funct
        if(token->type == TokenType::Name) {
            // TODO: add support for variables with statements, too
            new_statement.expressions.push_back(in_function_call());
        } else if(token->type != TokenType::End_Statement) {
            // Do nothing for now
        } else {
            // End of this statement; go back to prior state
            return new_statement;
        }
        ++token;
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
            m_functions.push_back(new_funct);
            return;
        }

        ++token;
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
            break;
        }

        ++token;
    }
}

void Parser::print_functions()
{
    for(const auto&[name, params, statements] : m_functions) {
        std::cout << "Function: " << name << "\n  Params: ";
        for(const auto &each : params) {
            std::cout << each << ' ';
        }
        std::cout << "\n  Statements: ";
        for(const auto &statemt : statements) {
            for(const auto &expr : statemt.expressions) {
                std::cout << expr.name << ' ';
                for(const auto arg : expr.arguments) {
                    std::cout << arg << ", ";
                }
            }
            std::cout << '|';
        }
        std::cout << '\n';
    }
}
