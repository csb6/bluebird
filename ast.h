#ifndef AST_CLASS_H
#define AST_CLASS_H
#include "token.h"
#include <CorradePointer.h>
#include <iosfwd>
#include <string>
#include <vector>
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

enum class TypeCategory : char {
    Range, Normal, Literal
};

// A lazily-evaluated sequence
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
    // Some default types that don't have to be declared
    static const Type Void;
    std::string name;
    Type() {}
    virtual ~Type() {}
    explicit Type(const std::string &n) : name(n) {}
    virtual TypeCategory category() const { return TypeCategory::Normal; }
    virtual void print(std::ostream&) const;
};

// "Typeless" literals (have no bounds, match with type of typed values
// within the same expression)
struct LiteralType : public Type {
    static const LiteralType String, Char, Int, Float;
    virtual TypeCategory category() const override { return TypeCategory::Literal; }
    using Type::Type;
};

// Type with integer bounds
struct RangeType : public Type {
    Range range;
    TypeCategory category() const override { return TypeCategory::Range; }
    using Type::Type;
    explicit RangeType(const std::string &n, Range &&r)
        : Type(n), range(r)
    {}
};

// An abstract object or non-standalone group of expressions
struct Expression {
    virtual ~Expression() {}
    // What kind of expression this is (e.g. a literal, lvalue, binary expr, etc.)
    virtual ExpressionType expr_type() const = 0;
    // What the type (in the language) this expression is. May be calculated when
    // called by visiting child nodes
    virtual const Type* type() const = 0;
    virtual void print(std::ostream&) const = 0;
};

// Each type of literal is a nameless instance of data
struct StringLiteral : public Expression {
    std::string value;
    explicit StringLiteral(const std::string& v) : value(v) {}
    const Type* type() const override { return &LiteralType::String; }
    ExpressionType expr_type() const override { return ExpressionType::StringLiteral; }
    void print(std::ostream&) const override;
};

struct CharLiteral : public Expression {
    char value;
    explicit CharLiteral(char v) : value(v) {}
    const Type* type() const override { return &LiteralType::Char; }
    ExpressionType expr_type() const override { return ExpressionType::CharLiteral; }
    void print(std::ostream&) const override;
};

struct IntLiteral : public Expression {
    // Holds arbitrarily-sized integers
    multi_int value;
    unsigned short bit_size;
    explicit IntLiteral(const std::string& v);
    const Type* type() const override { return &LiteralType::Int; }
    ExpressionType expr_type() const override { return ExpressionType::IntLiteral; }
    void print(std::ostream&) const override;
};

struct FloatLiteral : public Expression {
    double value;
    explicit FloatLiteral(int v) : value(v) {}
    const Type* type() const override { return &LiteralType::Float; }
    ExpressionType expr_type() const override { return ExpressionType::FloatLiteral; }
    void print(std::ostream&) const override;
};

// An expression consisting solely of an lvalue
struct LValueExpression : public Expression {
    std::string name;
    const struct LValue *lvalue;
    LValueExpression(const std::string &n,
                     const LValue *v) : name(n), lvalue(v) {}
    const Type* type() const override;
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
    const Type* type() const override { return right->type(); }
    void print(std::ostream&) const override;
};

// An expression that consists of an operator and two expressions
struct BinaryExpression : public Expression {
    Magnum::Pointer<Expression> left;
    TokenType op;
    Magnum::Pointer<Expression> right;

    BinaryExpression(Expression* l, TokenType oper, Expression* r);
    const Type* type() const override;
    ExpressionType expr_type() const override { return ExpressionType::Binary; }
    void print(std::ostream&) const override;
};

// A usage of a function
struct FunctionCall : public Expression {
    std::string name;
    std::vector<Magnum::Pointer<Expression>> arguments;
    struct Function* function;

    // TODO: have this be the return type
    const Type* type() const override { return &Type::Void; }
    ExpressionType expr_type() const override { return ExpressionType::FunctionCall; }
    void print(std::ostream&) const override;
};

// A named object that holds a value and can be assigned at least once
struct LValue {
    std::string name;
    RangeType* type;
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
#endif
