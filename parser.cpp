#include "parser.h"
#include <stack>
#include <iostream>

void print_error(unsigned int line_num, std::string_view message)
{
    std::cerr << "Line " << line_num << ": "
              << message << "\n";
}

enum class State : char {
    Start, InFunctDef, InFunctParams, InFunctBody, InStatement
};

Parser::Parser(std::vector<Token>::const_iterator input_begin,
               std::vector<Token>::const_iterator input_end)
    : m_input_begin(input_begin), m_input_end(input_end)
{}

void Parser::run()
{
    Function new_funct;
    Statement new_statement;

    std::stack<State> state_stack;
    State curr_state = State::Start;
    auto token = m_input_begin;
    while(token != m_input_end) {
        switch(curr_state) {
        case State::Start:
            switch(token->type) {
            case TokenType::Keyword_Funct:
                // Found start of a function definition
                curr_state = State::InFunctDef;
                break;
            case TokenType::Name:
                // Found beginning of some kind of statement/function call
                curr_state = State::InStatement;
                --token;
                break;
            default:
                break;
            }
            break;
        case State::InFunctDef:
            // First, set the function's name
            if(token->type == TokenType::Name) {
                new_funct.name = token->text;
            } else {
                print_error(token->line_num, "Expected name to follow `funct` keyword");
                exit(1);
            }

            ++token;
            if(token->type != TokenType::Open_Parentheses) {
                print_error(token->line_num,
                            "Expected `(` to follow name of function in definition");
                exit(1);
            }
            curr_state = State::InFunctParams;
            break;
        case State::InFunctParams:
            // TODO: add support for type annotations next to param names
            //      and commas after each name/type
            if(token->type == TokenType::Name) {
                new_funct.parameters.push_back(token->text);
            } else if(token->type == TokenType::Closed_Parentheses) {
                // Next, look for `is` keyword (TODO: open a new scope here)
                ++token;
                if(token->type != TokenType::Keyword_Is) {
                    print_error(token->line_num,
                                "Expected keyword `is` to follow parameters of the `funct`");
                    exit(1);
                }
                curr_state = State::InFunctBody;
            } else {
                print_error(token->line_num, "Expected parameter name");
                exit(1);
            }
            break;
        case State::InFunctBody:
            if(token->type == TokenType::End_Statement) {
                // Finished parsing a statement, time to add it as a child
                new_funct.statements.push_back(new_statement);
                new_statement = {};
            } else if(token->type != TokenType::Keyword_End) {
                // Start parsing this statement
                state_stack.push(curr_state);
                curr_state = State::InStatement;
                --token; // Put the statement's first token back
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
                new_funct = {};
                curr_state = State::Start;
            }
            break;
        case State::InStatement:
            // TODO: handle statements not inside a funct
            if(token->type != TokenType::End_Statement) {
                new_statement.expressions.push_back(token->text);
            } else {
                // End of this statement; go back to prior state
                curr_state = state_stack.top();
                state_stack.pop();
                --token; // Put the End_Statement token back
            }
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
                std::cout << expr << ' ';
            }
            std::cout << '|';
        }
        std::cout << '\n';
    }
}
