#ifndef AST_CLASS_H
#define AST_CLASS_H
#include "token.h"
#include <CorradePointer.h>
#include <type_traits>
#include <iomanip>
#include <iosfwd>
#include <string>
#include <vector>

namespace Magnum = Corrade::Containers;

enum class NameType : char {
    LValue, Funct, DeclaredFunct,
    Type, DeclaredType
};

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

    CompositeExpression(Expression* l, TokenType oper, Expression* r);
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
    virtual void print(std::ostream&) const = 0;
};

// A brief, usually one-line statement that holds a single expression
struct BasicStatement : public Statement {
    Magnum::Pointer<Expression> expression;
    void print(std::ostream& output) const override;
};

// Statement where a new variable is declared and optionally assigned the
// value of some expression
struct Initialization : public BasicStatement {
    LValue* target;
    void print(std::ostream& output) const override
    {
        target->print(output);
        BasicStatement::print(output);
    }
};

struct IfBlock : public Statement {
    Magnum::Pointer<Expression> condition;
    std::vector<Magnum::Pointer<Statement>> statements;
    void print(std::ostream& output) const override;
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
// Upper/lower bounds are inclusive
struct Range {
    long lower_bound, upper_bound;
    bool contains(long value) const;
};
#endif
