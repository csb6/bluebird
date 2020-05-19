#include "lexer.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <string_view>
#include <stack>

// An abstract object or non-standalone group of expressions
struct Expression {
    using Iterator = std::vector<Expression*>::iterator;
    virtual bool is_composite() const { return false; }
    virtual Iterator begin() { throw std::logic_error("begin() not implemented"); };
    virtual Iterator end() { throw std::logic_error("end() not implemented"); };
};

// A nameless piece of data
template<typename T>
struct Literal : public Expression {
    T value;
    explicit Literal(T v) : value(v) {}
};

struct Statement {
    std::vector<std::string> expressions;
};

// A procedure containing statements and optionally inputs/outputs
struct Function {
    std::string name;
    std::vector<std::string> parameters;

    std::vector<Statement> statements;
};

// A usage of a function
struct FunctionCall : public Expression {
    std::string function_name;
    std::vector<Expression*> arguments;

    bool is_composite() const override { return true; }

    Iterator begin() override
    {
        return arguments.begin();
    }

    Iterator end() override
    {
        return arguments.end();
    }
};


const std::string load_source_file(const char *filename)
{
    std::ifstream input_file(filename);
    return std::string{(std::istreambuf_iterator<char>(input_file)),
            (std::istreambuf_iterator<char>())};
}

enum class State : char {
    Start, InFunctDef, InFunctParams, InFunctBody, InStatement
};

void print_error(unsigned int line_num, std::string_view message)
{
    std::cerr << "Line " << line_num << ": "
              << message << "\n";
}

int main()
{
    const std::string source_file{load_source_file("lexer-test1.txt")};

    Lexer lexer{source_file.begin(), source_file.end()};
    lexer.run();
    //lexer.print_tokens();

    Function new_funct;
    Statement new_statement;

    std::vector<Function> functions;

    std::stack<State> state_stack;
    State curr_state = State::Start;
    auto token = lexer.begin();
    while(token != lexer.end()) {
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
                functions.push_back(new_funct);
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

    for(const auto&[name, params, statements] : functions) {
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

    return 0;
}
