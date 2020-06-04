#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include <vector>
#include <string>
#include "token.h"
#include <unordered_map>
#include <memory>

// An abstract object or non-standalone group of expressions
struct Expression {
    using Iterator = std::vector<std::unique_ptr<Expression>>::iterator;
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

// An expression that contains two or more other expressions, but
// is not a standalone function call statement
struct CompositeExpression : public Expression {
    std::vector<std::unique_ptr<Expression>> subexpressions;

    bool is_composite() const override { return true; }
    Iterator begin() override { return subexpressions.begin(); }
    Iterator end() override { return subexpressions.end(); }
};

// A usage of a function
struct FunctionCall : public Expression {
    std::string name;
    std::vector<std::unique_ptr<Expression>> arguments;

    bool is_composite() const override { return true; }
    Iterator begin() override { return arguments.begin(); }
    Iterator end() override { return arguments.end(); }
};

// A standalone piece of code terminated with a semicolon and consisting
// of one or more expressions
struct Statement {
    using Iterator = std::vector<std::unique_ptr<Expression>>::iterator;
    std::vector<std::unique_ptr<Expression>> expressions;
    Iterator begin() { return expressions.begin(); }
    Iterator end() { return expressions.end(); }
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
public:
    using TokenIterator = std::vector<Token>::const_iterator;
private:
    TokenIterator m_input_begin;
    TokenIterator m_input_end;
    TokenIterator token;
    std::vector<Function> m_functions;
    std::unordered_map<std::string, NameType> m_names_table;

    std::unique_ptr<Expression> in_literal();
    std::unique_ptr<FunctionCall> in_function_call();
    Statement in_statement();
    void in_function_definition();
public:
    Parser(TokenIterator input_begin,
           TokenIterator input_end);
    void run();
    void print_functions();
};
#endif
