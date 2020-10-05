#ifndef AST_CLASS_H
#define AST_CLASS_H
#include "token.h"
#include <CorradePointer.h>
#include <iosfwd>
#include <string>
#include <vector>
#include <string_view>
#include "multiprecision.h"

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

// An abstract object or non-standalone group of expressions
struct Expression {
    virtual ~Expression() {}
    // What kind of expression this is (e.g. a literal, lvalue, binary expr, etc.)
    virtual ExpressionType expr_type() const = 0;
    // What the type (in the language) this expression is. May be calculated when
    // called by visiting child nodes
    virtual std::string_view type() const = 0;
    virtual void print(std::ostream&) const = 0;
};

// Each type of literal is a nameless instance of data
struct StringLiteral : public Expression {
    std::string value;
    explicit StringLiteral(const std::string& v) : value(v) {}
    std::string_view type() const override { return "_string"; }
    ExpressionType expr_type() const override { return ExpressionType::StringLiteral; }
    void print(std::ostream&) const override;
};

struct CharLiteral : public Expression {
    char value;
    explicit CharLiteral(char v) : value(v) {}
    std::string_view type() const override { return "_char"; }
    ExpressionType expr_type() const override { return ExpressionType::CharLiteral; }
    void print(std::ostream&) const override;
};

struct IntLiteral : public Expression {
    // Holds arbitrarily-sized integers
    multi_int value;
    unsigned short bit_size;
    explicit IntLiteral(const std::string& v);
    std::string_view type() const override { return "_int"; }
    ExpressionType expr_type() const override { return ExpressionType::IntLiteral; }
    void print(std::ostream&) const override;
};

struct FloatLiteral : public Expression {
    double value;
    explicit FloatLiteral(int v) : value(v) {}
    std::string_view type() const override { return "_float"; }
    ExpressionType expr_type() const override { return ExpressionType::FloatLiteral; }
    void print(std::ostream&) const override;
};

// An expression consisting solely of an lvalue
struct LValueExpression : public Expression {
    std::string name;
    const struct LValue *lvalue;
    LValueExpression(const std::string &n,
                     const LValue *v) : name(n), lvalue(v) {}
    std::string_view type() const override;
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
    std::string_view type() const override { return right->type(); }
    void print(std::ostream&) const override;
};

// An expression that consists of an operator and two expressions
struct BinaryExpression : public Expression {
    Magnum::Pointer<Expression> left;
    TokenType op;
    Magnum::Pointer<Expression> right;

    BinaryExpression(Expression* l, TokenType oper, Expression* r);
    std::string_view type() const override { return left->type(); }
    ExpressionType expr_type() const override { return ExpressionType::Binary; }
    void print(std::ostream&) const override;
};

// A usage of a function
struct FunctionCall : public Expression {
    std::string name;
    std::vector<Magnum::Pointer<Expression>> arguments;
    struct Function* function;

    std::string_view type() const override { return "_no_type"; }
    ExpressionType expr_type() const override { return ExpressionType::FunctionCall; }
    void print(std::ostream&) const override;
};

// A named object that holds a value and can be assigned at least once
struct LValue {
    std::string name;
    struct RangeType* type;
    bool is_mutable = true;
    void print(std::ostream&) const;
};

// A standalone piece of code terminated with a semicolon and consisting
// of one or more expressions
struct Statement {
    Statement() {}
    explicit Statement(unsigned int line) : line_num(line) {}
    virtual ~Statement() {}
    unsigned int line_num;
    virtual StatementType type() const = 0;
    virtual void print(std::ostream&) const = 0;
};

// A brief, usually one-line statement that holds a single expression
struct BasicStatement : public Statement {
    Magnum::Pointer<Expression> expression;
    explicit BasicStatement(unsigned int line) : Statement(line) {}
    StatementType type() const override { return StatementType::Basic; }
    void print(std::ostream&) const override;
};

// Statement where a new variable is declared and optionally assigned the
// value of some expression
struct Initialization : public BasicStatement {
    LValue* lvalue;
    explicit Initialization(unsigned int line) : BasicStatement(line) {}
    StatementType type() const override { return StatementType::Initialization; }
    void print(std::ostream& output) const override;
};

struct IfBlock : public Statement {
    Magnum::Pointer<Expression> condition;
    std::vector<Statement*> statements;
    IfBlock(unsigned int line) : Statement(line) {}
    StatementType type() const override { return StatementType::IfBlock; }
    void print(std::ostream&) const override;
};

// A procedure containing statements and optionally inputs/outputs
struct Function {
    std::string name;
    std::vector<LValue*> parameters;
    std::vector<Statement*> statements;
    friend std::ostream& operator<<(std::ostream&, const Function&);
};

// A lazily-evaluated sequence of number-like objects
// Upper/lower bounds are inclusive
struct Range {
    multi_int lower_bound, upper_bound;
    unsigned short bit_size;
    Range() : bit_size(0) {}
    // Move Constructors
    Range(const multi_int& lower, const multi_int& upper);
};

// A kind of object
struct Type {
    std::string name;
    friend std::ostream& operator<<(std::ostream&, const Type&);
};

// Type with integer bounds
struct RangeType : public Type {
    Range range;
    //friend std::ostream& operator<<(std::ostream&, const RangeType&);
};
#endif
