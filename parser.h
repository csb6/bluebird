#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include <vector>
#include <string>
#include "token.h"
#include <unordered_map>
#include <memory>
#include <iosfwd>
#include <iomanip>
#include <type_traits>

// An abstract object or non-standalone group of expressions
struct Expression {
    virtual ~Expression() {}
    virtual bool is_composite() const { return false; }
    virtual void print(std::ostream&) const {};
};

// A nameless instance of data
template<typename T>
struct Literal : public Expression {
    T value;
    explicit Literal(T v) : value(v) {}
    void print(std::ostream& output) const override
    {
        if constexpr (std::is_floating_point_v<T>) {
            output << std::setprecision(10);
        }
        output << value;
    }
};

// An expression that contains two or more other expressions, but
// is not itself function call
struct CompositeExpression : public Expression {
    std::unique_ptr<Expression> left;
    TokenType op;
    std::unique_ptr<Expression> right;

    CompositeExpression(std::unique_ptr<Expression>&& l, TokenType oper,
                        std::unique_ptr<Expression>&& r);
    bool is_composite() const override { return true; }
    void print(std::ostream&) const override;
};

// A usage of a function
struct FunctionCall : public Expression {
    std::string name;
    std::vector<std::unique_ptr<Expression>> arguments;

    bool is_composite() const override { return true; }
    void print(std::ostream&) const override;
};

// A named object that holds a value and can be assigned at least once 
struct LValue {
    std::string name;
    std::string type;
    virtual ~LValue() {}
    virtual bool is_mutable() const { return true; }
    virtual void print(std::ostream&) const = 0;
};

// An lvalue that can be assigned a value more than once
struct Variable : public LValue {
    void print(std::ostream&) const override;
};

// An lvalue that can be assigned a value at most once
struct Constant : public LValue {
    bool is_mutable() const override { return false; }
    void print(std::ostream& output) const override
    {
        output << "Constant: " << name;
    }
};

// A standalone piece of code terminated with a semicolon and consisting
// of one or more expressions
struct Statement {
    virtual ~Statement() {}
    std::unique_ptr<Expression> expression;
    virtual void print(std::ostream&) const;
};

// Statement where a variable is assigned to the value of some expression
struct Assignment : public Statement {
    LValue* target;
    void print(std::ostream& output) const override
    {
        target->print(output);
        Statement::print(output);
    }
};

// A procedure containing statements and optionally inputs/outputs
struct Function {
    std::string name;
    std::vector<std::string> parameters;
    std::vector<std::unique_ptr<Statement>> statements;
    friend std::ostream& operator<<(std::ostream&, const Function&);
};

enum class NameType : char {
    LValue, Funct
};

class Parser {
public:
    using TokenIterator = std::vector<Token>::const_iterator;
private:
    TokenIterator m_input_begin;
    TokenIterator m_input_end;
    TokenIterator token;
    std::vector<Function> m_functions;
    std::vector<std::unique_ptr<LValue>> m_lvalues;
    std::unordered_map<std::string, NameType> m_names_table;

    // Handle each type of expression
    std::unique_ptr<Expression> in_literal();
    std::unique_ptr<Expression> in_multiply_divide();
    std::unique_ptr<Expression> in_add_subtract();
    std::unique_ptr<Expression> in_basic_expression();
    std::unique_ptr<Expression> in_composite_expression();
    std::unique_ptr<FunctionCall> in_function_call();
    std::unique_ptr<Expression> in_parentheses();
    std::unique_ptr<Expression> in_expression();
    // Handle each type of statement
    std::unique_ptr<Assignment> in_assignment();
    std::unique_ptr<Statement> in_statement();
    // Handle each function definition
    void in_function_definition();
public:
    Parser(TokenIterator input_begin,
           TokenIterator input_end);
    void run();
    friend std::ostream& operator<<(std::ostream&, const Parser&);
};
#endif
