/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020  Cole Blakley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef AST_CLASS_H
#define AST_CLASS_H
#include "token.h"
#include <CorradePointer.h>
#include <iosfwd>
#include <string>
#include <vector>
#include "multiprecision.h"

namespace Magnum = Corrade::Containers;

namespace llvm {
    class Value;
}

enum class NameType : char {
    LValue, Funct, DeclaredFunct,
    Type, DeclaredType
};

// Used in place of RTTI for differentiating between actual types of Expression*'s
enum class ExpressionKind : char {
    StringLiteral, CharLiteral, IntLiteral, FloatLiteral,
    LValue, Binary, Unary, FunctionCall
};

enum class StatementKind : char {
    Basic, Initialization, Assignment, IfBlock
};

enum class TypeCategory : char {
    Range, Normal, Literal
};

enum class ContextKind : char {
    None, Expression, LValue
};

// A lazily-evaluated sequence
// Upper/lower bounds are inclusive
struct Range {
    multi_int lower_bound, upper_bound;
    unsigned short bit_size;
    bool is_signed;

    Range() : bit_size(0), is_signed(true) {}
    Range(const multi_int& lower, const multi_int& upper);

    bool contains(const multi_int&) const;
};

// A kind of object
struct Type {
    // Some default types that don't have to be declared
    static const Type Void, String, Char, Int, Float;
    std::string name;

    Type() {}
    explicit Type(const std::string &n) : name(n) {}
    virtual ~Type() {}

    virtual unsigned short bit_size() const { return 0; }
    virtual TypeCategory   category() const { return TypeCategory::Normal; }
    virtual void           print(std::ostream&) const;
};

// Type with integer bounds
struct RangeType : public Type {
    Range range;

    using Type::Type;
    RangeType(const std::string &n, Range &&r) : Type(n), range(r) {}

    unsigned short bit_size() const override { return range.bit_size; }
    TypeCategory   category() const override { return TypeCategory::Range; }
    bool           is_signed() const { return range.is_signed; }
};

// An abstract object or non-standalone group of expressions
struct Expression {
    virtual ~Expression() {}

    // What kind of expression this is (e.g. a literal, lvalue, binary expr, etc.)
    virtual ExpressionKind kind() const = 0;
    // What the type (in the language) this expression is. May be calculated when
    // called by visiting child nodes
    virtual const Type*    type() const = 0;
    virtual void           print(std::ostream&) const = 0;

    // Used in typechecking. Definitions/overrides defined in checker.cpp
    virtual void         check_types(const struct Statement*) const = 0;
    // Used in code generation. Definitions/overrides defined in codegenerator.cpp
    virtual llvm::Value* codegen(class CodeGenerator&) = 0;
};

// Each type of literal is a nameless instance of data
struct StringLiteral final : public Expression {
    std::string value;

    explicit StringLiteral(const std::string& v) : value(v) {}

    ExpressionKind kind() const override { return ExpressionKind::StringLiteral; }
    const Type*    type() const override { return &Type::String; }
    void           print(std::ostream&) const override;

    void         check_types(const Statement*) const override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct CharLiteral final : public Expression {
    char value;

    explicit CharLiteral(char v) : value(v) {}

    ExpressionKind kind() const override { return ExpressionKind::CharLiteral; }
    const Type*    type() const override { return &Type::Char; }
    void           print(std::ostream&) const override;

    void         check_types(const Statement*) const override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct IntLiteral final : public Expression {
    // Holds arbitrarily-sized integers
    multi_int value;
    ContextKind context_kind = ContextKind::None;
    union {
        // The expression that determines the type of this literal
        Expression* context_expr = nullptr;
        struct LValue* context_lvalue;
    };

    explicit IntLiteral(const std::string& v) : value(v) {}
    explicit IntLiteral(const multi_int& v) : value(v) {}

    ExpressionKind kind() const override { return ExpressionKind::IntLiteral; }
    const Type*    type() const override;
    void           print(std::ostream&) const override;

    void         check_types(const Statement*) const override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct FloatLiteral final : public Expression {
    double value;

    explicit FloatLiteral(double v) : value(v) {}

    ExpressionKind kind() const override { return ExpressionKind::FloatLiteral; }
    const Type*    type() const override { return &Type::Float; }
    void           print(std::ostream&) const override;

    void         check_types(const Statement*) const override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

// An expression consisting solely of an lvalue
struct LValueExpression final : public Expression {
    std::string name;
    const struct LValue *lvalue;

    LValueExpression(const std::string &n, const LValue *v)
        : name(n), lvalue(v) {}

    ExpressionKind kind() const override { return ExpressionKind::LValue; }
    const Type*    type() const override;
    // Other data should be looked up in the corresponding LValue object
    void           print(std::ostream&) const override;

    void         check_types(const Statement*) const override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

// An expression that consists of an operator and an expression
struct UnaryExpression final : public Expression {
    TokenType op;
    Magnum::Pointer<Expression> right;

    UnaryExpression(TokenType, Expression*);

    ExpressionKind kind() const override { return ExpressionKind::Unary; }
    const Type*    type() const override { return right->type(); }
    void           print(std::ostream&) const override;

    void         check_types(const Statement*) const override;
    llvm::Value* codegen(CodeGenerator&) override;
};

// An expression that consists of an operator and two expressions
struct BinaryExpression final : public Expression {
    Magnum::Pointer<Expression> left;
    TokenType op;
    Magnum::Pointer<Expression> right;

    BinaryExpression(Expression* l, TokenType oper, Expression* r);

    ExpressionKind kind() const override { return ExpressionKind::Binary; }
    const Type*    type() const override;
    void           print(std::ostream&) const override;

    void         check_types(const Statement*) const override;
    llvm::Value* codegen(CodeGenerator&) override;
};

// A usage of a function
struct FunctionCall final : public Expression {
    std::string name;
    std::vector<Magnum::Pointer<Expression>> arguments;
    struct Function* function;

    explicit FunctionCall(const std::string& name) : name(name) {}

    // TODO: have this be the return type
    ExpressionKind kind() const override { return ExpressionKind::FunctionCall; }
    const Type*    type() const override { return &Type::Void; }
    void           print(std::ostream&) const override;

    void         check_types(const Statement*) const override;
    llvm::Value* codegen(CodeGenerator&) override;
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
    unsigned int line_num;

    explicit Statement(unsigned int line) : line_num(line) {}
    virtual ~Statement() {}

    virtual StatementKind kind() const = 0;
    virtual void          print(std::ostream&) const = 0;
    virtual void          check_types() const = 0;
};

// A brief, usually one-line statement that holds a single expression
struct BasicStatement final : public Statement {
    Magnum::Pointer<Expression> expression;

    BasicStatement(unsigned int line, Expression* expr)
        : Statement(line), expression(expr) {}

    StatementKind kind() const override { return StatementKind::Basic; }
    void          print(std::ostream&) const override;
    void          check_types() const override;
};

// Statement where a new variable is declared and optionally assigned the
// value of some expression
struct Initialization final : public Statement {
    Magnum::Pointer<Expression> expression{nullptr};
    LValue* lvalue;

    Initialization(unsigned int line, LValue* lval)
        : Statement(line), lvalue(lval) {}

    StatementKind kind() const override { return StatementKind::Initialization; }
    void          print(std::ostream& output) const override;
    void          check_types() const override;
};

struct Assignment final : public Statement {
    Magnum::Pointer<Expression> expression;
    LValue* lvalue;

    Assignment(unsigned line, Expression* expr, LValue* lv)
        : Statement(line), expression(expr), lvalue(lv) {}
    StatementKind kind() const override { return StatementKind::Assignment; }
    void          print(std::ostream& output) const override;
    void          check_types() const override;
};

struct IfBlock final : public Statement {
    Magnum::Pointer<Expression> condition;
    std::vector<Statement*> statements;

    IfBlock(unsigned int line, Expression* cond)
        : Statement(line), condition(cond) {}

    StatementKind kind() const override { return StatementKind::IfBlock; }
    void          print(std::ostream&) const override;
    void          check_types() const override;
};

// A procedure containing statements and optionally inputs/outputs
struct Function {
    std::string name;
    std::vector<LValue*> parameters;
    std::vector<Statement*> statements;

    explicit Function(const std::string& n) : name(n) {}

    friend std::ostream& operator<<(std::ostream&, const Function&);
};
#endif
