#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include <vector>
#include <string>
#include "token.h"
#include <unordered_map>

// An abstract object or non-standalone group of expressions
struct Expression {
    using Iterator = std::vector<Expression*>::iterator;
    virtual ~Expression() {}
    virtual bool is_composite() const { return false; }
    virtual Iterator begin() { throw std::logic_error("begin() not implemented"); };
    virtual Iterator end() { throw std::logic_error("end() not implemented"); };
};

// A nameless piece of data
template<typename T>
struct Literal : public Expression {
    T value;
    explicit Literal(T v) : value(v) {}
    ~Literal() {}
};

// A usage of a function
struct FunctionCall : public Expression {
    std::string name;
    std::vector<std::string> arguments;
    ~FunctionCall() {}

    //bool is_composite() const override { return true; }
    //Iterator begin() override { return arguments.begin(); }
    //Iterator end() override { return arguments.end(); }
};

struct Statement {
    std::vector<FunctionCall> expressions;
};

// A procedure containing statements and optionally inputs/outputs
struct Function {
    std::string name;
    std::vector<std::string> parameters;

    std::vector<Statement> statements;
};

enum class NameType : char {
    Variable, Funct
};

class Parser {
private:
    std::vector<Function> m_functions;
    std::vector<Token>::const_iterator m_input_begin;
    std::vector<Token>::const_iterator m_input_end;
    std::unordered_map<std::string, NameType> m_names_table;
public:
    Parser(std::vector<Token>::const_iterator input_begin,
           std::vector<Token>::const_iterator input_end);
    void run();
    void print_functions();
};
#endif
