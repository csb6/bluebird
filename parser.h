#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include <vector>
#include <string>
#include "token.h"

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
    Iterator begin() override { return arguments.begin(); }
    Iterator end() override { return arguments.end(); }
};

class Parser {
private:
    std::vector<Function> m_functions;
    std::vector<Token>::const_iterator m_input_begin;
    std::vector<Token>::const_iterator m_input_end;
public:
    Parser(std::vector<Token>::const_iterator input_begin,
           std::vector<Token>::const_iterator input_end);
    void run();
    void print_functions();
};
#endif
