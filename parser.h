#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include <vector>
#include <string>
#include "token.h"
#include <unordered_map>
#include <CorradePointer.h>
#include <iosfwd>
#include <iomanip>
#include <type_traits>

namespace Magnum = Corrade::Containers;

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

// An expression consisting solely of an lvalue
struct LValueExpression : public Expression {
    std::string name;
    // Other data should be looked up in the corresponding
    // LValue object
    void print(std::ostream& output) const override;
};

// An expression that contains two or more other expressions, but
// is not itself function call
struct CompositeExpression : public Expression {
    Magnum::Pointer<Expression> left;
    TokenType op;
    Magnum::Pointer<Expression> right;

    CompositeExpression(Magnum::Pointer<Expression>&& l, TokenType oper,
                        Magnum::Pointer<Expression>&& r);
    bool is_composite() const override { return true; }
    void print(std::ostream&) const override;
};

// A usage of a function
struct FunctionCall : public Expression {
    std::string name;
    std::vector<Magnum::Pointer<Expression>> arguments;

    bool is_composite() const override { return true; }
    void print(std::ostream&) const override;
};

// A named object that holds a value and can be assigned at least once 
struct LValue {
    std::string name;
    std::string type;
    bool is_mutable = true;
    void print(std::ostream&) const;
};

// A standalone piece of code terminated with a semicolon and consisting
// of one or more expressions
struct Statement {
    virtual ~Statement() {}
    Magnum::Pointer<Expression> expression;
    virtual void print(std::ostream&) const;
};

// TODO: add way to declare an lvalue without assigning initial value
// Statement where a new variable is declared and assigned the
// value of some expression
struct Initialization : public Statement {
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
    std::vector<Magnum::Pointer<LValue>> parameters;
    std::vector<Magnum::Pointer<Statement>> statements;
    friend std::ostream& operator<<(std::ostream&, const Function&);
};

// A kind of object
struct Type {
    std::string name;
    friend std::ostream& operator<<(std::ostream&, const Type&);
};

// A lazily-evaluated sequence of number-like objects
// Upped/lower bounds are inclusive
struct Range {
    long lower_bound, upper_bound;
    bool contains(long value) const;
};


enum class NameType : char {
    LValue, Funct, DeclaredFunct,
    Type, DeclaredType
};

class Parser {
public:
    using TokenIterator = std::vector<Token>::const_iterator;
private:
    TokenIterator m_input_begin;
    TokenIterator m_input_end;
    TokenIterator token;
    std::vector<Function> m_functions;
    std::vector<Type> m_types;
    std::vector<Magnum::Pointer<LValue>> m_lvalues;
    std::unordered_map<std::string, NameType> m_names_table;

    Magnum::Pointer<Expression> parse_expression(TokenType right_bind_power = {});
    // Helpers
    Magnum::Pointer<LValue> in_lvalue_declaration();
    // Handle each type of expression
    Expression* in_literal();
    Expression* in_lvalue_expression();
    Expression* in_basic_expression();
    FunctionCall* in_function_call();
    // Handle each type of statement
    Magnum::Pointer<Initialization> in_initialization();
    Magnum::Pointer<Statement> in_statement();
    // Handle each function definition
    void in_function_definition();
    // Types
    void in_type_definition();
public:
    Parser(TokenIterator input_begin,
           TokenIterator input_end);
    void run();
    friend std::ostream& operator<<(std::ostream&, const Parser&);
};
#endif
