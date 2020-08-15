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

// Used in place of RTTI for differentiating between actual types of Expression*'s
enum class ExpressionType : char {
    StringLiteral, CharLiteral, IntLiteral, FloatLiteral,
    LValue, Binary, FunctionCall
};

enum class StatementType : char {
    Basic, Initialization, IfBlock
};

// An abstract object or non-standalone group of expressions
struct Expression {
    virtual ~Expression() {}
    virtual ExpressionType type() const = 0;
    virtual void print(std::ostream&) const = 0;
};

// These typedefs used in codegenerator.cpp to keep the T type in Literal<T>
// in sync with the casting/representation in the codegen stage
using StringLiteral_t = std::string;
using CharLiteral_t = char;
using IntLiteral_t = int;
using FloatLiteral_t = double;
// A nameless instance of data
template<typename T>
struct Literal : public Expression {
    T value;
    explicit Literal(T v) : value(v) {}
    ExpressionType type() const override
    {
        if constexpr (std::is_same_v<T, StringLiteral_t>) {
            return ExpressionType::StringLiteral;
        } else if (std::is_same_v<T, CharLiteral_t>) {
            return ExpressionType::CharLiteral;
        } else if (std::is_same_v<T, IntLiteral_t>) {
            return ExpressionType::IntLiteral;
        } else if (std::is_same_v<T, FloatLiteral_t>) {
            return ExpressionType::FloatLiteral;
        }
    }

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
    ExpressionType type() const override { return ExpressionType::LValue; }
    // Other data should be looked up in the corresponding
    // LValue object
    void print(std::ostream& output) const override;
};

// An expression that consists of an operator and two expressions
struct BinaryExpression : public Expression {
    Magnum::Pointer<Expression> left;
    TokenType op;
    Magnum::Pointer<Expression> right;

    BinaryExpression(Expression* l, TokenType oper, Expression* r);
    ExpressionType type() const override { return ExpressionType::Binary; }
    void print(std::ostream&) const override;
};

// A usage of a function
struct FunctionCall : public Expression {
    std::string name;
    std::vector<Magnum::Pointer<Expression>> arguments;

    ExpressionType type() const override { return ExpressionType::FunctionCall; }
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
    virtual StatementType type() const = 0;
    virtual void print(std::ostream&) const = 0;
};

// A brief, usually one-line statement that holds a single expression
struct BasicStatement : public Statement {
    Magnum::Pointer<Expression> expression;
    StatementType type() const override { return StatementType::Basic; }
    void print(std::ostream& output) const override;
};

// Statement where a new variable is declared and optionally assigned the
// value of some expression
struct Initialization : public BasicStatement {
    LValue* target;
    StatementType type() const override { return StatementType::Initialization; }
    void print(std::ostream& output) const override
    {
        target->print(output);
        BasicStatement::print(output);
    }
};

struct IfBlock : public Statement {
    Magnum::Pointer<Expression> condition;
    std::vector<Magnum::Pointer<Statement>> statements;
    StatementType type() const override { return StatementType::IfBlock; }
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
