#ifndef AST_CLASS_H
#define AST_CLASS_H
#include "token.h"
#include <CorradePointer.h>
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
    LValue, Binary, Unary, FunctionCall
};

enum class StatementType : char {
    Basic, Initialization, IfBlock
};

// Represents a unique type (e.g. Number, Positive, String) or symbol (e.g. a variable)
using SymbolId = unsigned short;
// Basic, "unconstrained" types (used for typeless constants, literals, etc.)
constexpr SymbolId NoType = 0;
constexpr SymbolId StringType = 1;
constexpr SymbolId CharType = 2;
constexpr SymbolId IntType = 3;
constexpr SymbolId FloatType = 4;
constexpr SymbolId FirstFreeId = 5;

// An abstract object or non-standalone group of expressions
struct Expression {
    virtual ~Expression() {}
    // What kind of expression this is (e.g. a literal, lvalue, binary expr, etc.)
    virtual ExpressionType expr_type() const = 0;
    // What the type (in the language) this expression is. May be calculated when
    // called by visiting child nodes
    virtual SymbolId type() const = 0;
    virtual void print(std::ostream&) const = 0;
};

// Each type of literal is a nameless instance of data
struct StringLiteral : public Expression {
    std::string value;
    explicit StringLiteral(const std::string& v) : value(v) {}
    SymbolId type() const override { return StringType; }
    ExpressionType expr_type() const override { return ExpressionType::StringLiteral; }
    void print(std::ostream&) const override;
};

struct CharLiteral : public Expression {
    char value;
    explicit CharLiteral(char v) : value(v) {}
    SymbolId type() const override { return CharType; }
    ExpressionType expr_type() const override { return ExpressionType::CharLiteral; }
    void print(std::ostream&) const override;
};

struct IntLiteral : public Expression {
    int value;
    explicit IntLiteral(int v) : value(v) {}
    SymbolId type() const override { return IntType; }
    ExpressionType expr_type() const override { return ExpressionType::IntLiteral; }
    void print(std::ostream&) const override;
};

struct FloatLiteral : public Expression {
    double value;
    explicit FloatLiteral(int v) : value(v) {}
    SymbolId type() const override { return FloatType; }
    ExpressionType expr_type() const override { return ExpressionType::FloatLiteral; }
    void print(std::ostream&) const override;
};

// An expression consisting solely of an lvalue
struct LValueExpression : public Expression {
    std::string name;
    SymbolId type() const override { return NoType; } //TODO: implement
    ExpressionType expr_type() const override { return ExpressionType::LValue; }
    // Other data should be looked up in the corresponding LValue object
    void print(std::ostream&) const override;
};

// An expression that consists of an operator and an expression
struct UnaryExpression : public Expression {
    TokenType op;
    Magnum::Pointer<Expression> right;

    UnaryExpression(TokenType, Expression*);
    ExpressionType expr_type() const override { return ExpressionType::Unary; }
    SymbolId type() const override { return right->type(); }
    void print(std::ostream&) const override;
};

// An expression that consists of an operator and two expressions
struct BinaryExpression : public Expression {
    Magnum::Pointer<Expression> left;
    TokenType op;
    Magnum::Pointer<Expression> right;

    BinaryExpression(Expression* l, TokenType oper, Expression* r);
    SymbolId type() const override { return left->type(); }
    ExpressionType expr_type() const override { return ExpressionType::Binary; }
    void print(std::ostream&) const override;
};

// A usage of a function
struct FunctionCall : public Expression {
    std::string name;
    std::vector<Magnum::Pointer<Expression>> arguments;
    SymbolId return_type = NoType;

    SymbolId type() const override { return return_type; }
    ExpressionType expr_type() const override { return ExpressionType::FunctionCall; }
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
    explicit Statement(unsigned int line) : line_num(line) {}
    virtual ~Statement() {}
    unsigned int line_num;
    virtual StatementType type() const = 0;
    virtual void print(std::ostream&) const = 0;
};

// A brief, usually one-line statement that holds a single expression
struct BasicStatement : public Statement {
    Magnum::Pointer<Expression> expression;
    BasicStatement(unsigned int line) : Statement(line) {}
    StatementType type() const override { return StatementType::Basic; }
    void print(std::ostream&) const override;
};

// Statement where a new variable is declared and optionally assigned the
// value of some expression
struct Initialization : public BasicStatement {
    LValue* target;
    Initialization(unsigned int line) : BasicStatement(line) {}
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
    IfBlock(unsigned int line) : Statement(line) {}
    StatementType type() const override { return StatementType::IfBlock; }
    void print(std::ostream&) const override;
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
