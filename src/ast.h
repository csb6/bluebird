/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2021  Cole Blakley

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
    StringLiteral, CharLiteral, IntLiteral, BoolLiteral, FloatLiteral,
    LValue, Binary, Unary, FunctionCall
};

enum class StatementKind : char {
    Basic, Initialization, Assignment, IfBlock, Block, While, Return
};

enum class TypeCategory : char {
    Range, Normal, Literal, Boolean
};

enum class FunctionKind : char {
    Normal, Builtin
};

// A lazily-evaluated sequence
// Upper/lower bounds are inclusive
struct Range {
    multi_int lower_bound, upper_bound;
    bool is_signed;
    unsigned short bit_size;

    Range() : is_signed(true), bit_size(0) {}
    Range(const multi_int& lower, const multi_int& upper);

    bool contains(const multi_int&) const;
};
std::ostream& operator<<(std::ostream& output, const Range&);

// A kind of object
struct Type {
    // Some default types that don't have to be declared
    static const Type Void, String, Float;
    std::string name;

    Type() = default;
    explicit Type(const std::string &n) : name(n) {}
    virtual ~Type() noexcept = default;

    // TODO: fix bug where this is called for some boolean literals
    //  (Type::bit_size should never be called)
    virtual unsigned short bit_size() const { return 1; }
    virtual TypeCategory   category() const { return TypeCategory::Normal; }
    virtual void           print(std::ostream&) const;
};

struct LiteralType final : public Type {
    static const LiteralType Char, Int, Bool;

    using Type::Type;

    TypeCategory category() const override { return TypeCategory::Literal; }
};

// Type with limited set of named, enumerated values
struct EnumType final : public Type {
    static EnumType Boolean;
    // TODO: change kind when non-bool enum types added
    TypeCategory kind = TypeCategory::Boolean;

    using Type::Type;

    unsigned short bit_size() const override { return 1; }
    TypeCategory   category() const override { return kind; }
};

// Type with integer bounds
struct RangeType final : public Type {
    // Some more default types that don't have to be declared
    static RangeType Integer, Character;
    Range range;

    using Type::Type;
    RangeType(const std::string &n,
              const multi_int& lower_limit,
              const multi_int& upper_limit)
        : Type(n), range(lower_limit, upper_limit) {}

    unsigned short bit_size() const override { return range.bit_size; }
    TypeCategory   category() const override { return TypeCategory::Range; }
    void           print(std::ostream&) const override;
    bool           is_signed() const { return range.is_signed; }
};

// An abstract object or non-standalone group of expressions
struct Expression {
    virtual ~Expression() noexcept = default;

    // What kind of expression this is (e.g. a literal, lvalue, binary expr, etc.)
    virtual ExpressionKind kind() const = 0;
    // What the type (in the language) this expression is. May be calculated when
    // called by visiting child nodes
    virtual const Type*    type() const = 0;
    virtual unsigned int   line_num() const = 0;
    virtual void           print(std::ostream&) const = 0;

    // Used in typechecking. Definitions/overrides defined in checker.cpp
    virtual void         check_types() = 0;
    // Used in code generation. Definitions/overrides defined in codegenerator.cpp
    virtual llvm::Value* codegen(class CodeGenerator&) = 0;
};

// Each type of literal is a nameless instance of data
struct StringLiteral final : public Expression {
    std::string value;
    unsigned int line;

    StringLiteral(unsigned int line_n, const std::string& v)
        : value(v), line(line_n) {}

    ExpressionKind kind() const override { return ExpressionKind::StringLiteral; }
    const Type*    type() const override { return &Type::String; }
    unsigned int   line_num() const override { return line; }
    void           print(std::ostream&) const override;

    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct CharLiteral final : public Expression {
    char value;
    unsigned int line;
    const Type* actual_type = &LiteralType::Char;

    CharLiteral(unsigned int line_n, char v) : value(v), line(line_n) {}

    ExpressionKind kind() const override { return ExpressionKind::CharLiteral; }
    const Type*    type() const override { return actual_type; }
    unsigned int   line_num() const override { return line; }
    void           print(std::ostream&) const override;

    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct IntLiteral final : public Expression {
    // Holds arbitrarily-sized integers
    multi_int value;
    unsigned int line;
    const Type* actual_type = &LiteralType::Int;

    IntLiteral(unsigned int line_n, const std::string& v)
        : value(v), line(line_n) {}
    IntLiteral(unsigned int line_n, const multi_int& v)
        : value(v), line(line_n) {}

    ExpressionKind kind() const override { return ExpressionKind::IntLiteral; }
    const Type*    type() const override { return actual_type; }
    unsigned int   line_num() const override { return line; }
    void           print(std::ostream&) const override;

    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct BoolLiteral final : public Expression {
    bool value;
    unsigned int line;
    const Type* actual_type = &LiteralType::Bool;

    BoolLiteral(unsigned int line_n, bool v) : value(v), line(line_n) {}

    ExpressionKind kind() const override { return ExpressionKind::BoolLiteral; }
    const Type*    type() const override { return actual_type; }
    unsigned int   line_num() const override { return line; }
    void           print(std::ostream&) const override;

    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct FloatLiteral final : public Expression {
    double value;
    unsigned int line;

    FloatLiteral(unsigned int line_n, double v) : value(v), line(line_n) {}

    ExpressionKind kind() const override { return ExpressionKind::FloatLiteral; }
    const Type*    type() const override { return &Type::Float; }
    unsigned int   line_num() const override { return line; }
    void           print(std::ostream&) const override;

    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

// An expression consisting solely of an lvalue
struct LValueExpression final : public Expression {
    std::string name;
    const struct LValue *lvalue;
    unsigned int line;

    LValueExpression(unsigned int line_n, const std::string &n, const LValue *v)
        : name(n), lvalue(v), line(line_n) {}

    ExpressionKind kind() const override { return ExpressionKind::LValue; }
    const Type*    type() const override;
    unsigned int   line_num() const override { return line; }
    // Other data should be looked up in the corresponding LValue object
    void           print(std::ostream&) const override;

    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

// An expression that consists of an operator and an expression
struct UnaryExpression final : public Expression {
    TokenType op;
    Magnum::Pointer<Expression> right;

    UnaryExpression(TokenType, Expression*);

    ExpressionKind kind() const override { return ExpressionKind::Unary; }
    const Type*    type() const override { return right->type(); }
    unsigned int   line_num() const override { return right->line_num(); }
    void           print(std::ostream&) const override;

    void         check_types() override;
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
    unsigned int   line_num() const override { return left->line_num(); }
    void           print(std::ostream&) const override;

    void         check_types() override;
    llvm::Value* codegen(CodeGenerator&) override;
};

// A usage of a function
struct FunctionCall final : public Expression {
    std::string name;
    std::vector<Magnum::Pointer<Expression>> arguments;
    struct Function* function;
    unsigned int line;

    FunctionCall(unsigned int line_n, const std::string& name)
        : name(name), line(line_n) {}

    // TODO: have this be the return type
    ExpressionKind kind() const override { return ExpressionKind::FunctionCall; }
    const Type*    type() const override;
    unsigned int   line_num() const override { return line; }
    void           print(std::ostream&) const override;

    void         check_types() override;
    llvm::Value* codegen(CodeGenerator&) override;
};

// A named object that holds a value and can be assigned at least once
struct LValue {
    std::string name;
    Type* type;
    bool is_mutable = true;

    explicit LValue(const std::string& n) : name(n) {}
    LValue(const std::string& n, Type* t) : name(n), type(t) {}

    void print(std::ostream&) const;
};

// A standalone piece of code terminated with a semicolon and consisting
// of one or more expressions
struct Statement {
    virtual ~Statement() noexcept = default;

    virtual StatementKind kind() const = 0;
    virtual unsigned int  line_num() const = 0;
    virtual void          print(std::ostream&) const = 0;
    // Returns true if always returns, no matter code path
    virtual bool          check_types(class Checker&) = 0;
};

// A brief, usually one-line statement that holds a single expression
struct BasicStatement final : public Statement {
    Magnum::Pointer<Expression> expression;

    explicit BasicStatement(Expression* expr) : expression(expr) {}

    StatementKind kind() const override { return StatementKind::Basic; }
    unsigned int  line_num() const override { return expression->line_num(); }
    void          print(std::ostream&) const override;
    bool          check_types(Checker&) override;
};

// Statement where a new variable is declared and optionally assigned the
// value of some expression
struct Initialization final : public Statement {
    Magnum::Pointer<Expression> expression{nullptr};
    Magnum::Pointer<LValue> lvalue;

    explicit Initialization(LValue* lval) : lvalue(lval) {}

    StatementKind kind() const override { return StatementKind::Initialization; }
    unsigned int  line_num() const override { return expression->line_num(); }
    void          print(std::ostream& output) const override;
    bool          check_types(Checker&) override;
};

// Statement where an existing variable is given a value
struct Assignment final : public Statement {
    Magnum::Pointer<Expression> expression;
    LValue* lvalue;

    Assignment(Expression* expr, LValue* lv) : expression(expr), lvalue(lv) {}

    StatementKind kind() const override { return StatementKind::Assignment; }
    unsigned int  line_num() const override { return expression->line_num(); }
    void          print(std::ostream& output) const override;
    bool          check_types(Checker&) override;
};

// A group of statements contained in a scope
struct Block : public Statement {
    std::vector<Magnum::Pointer<Statement>> statements;
    unsigned int line;

    explicit Block(unsigned int l) : line(l) {}

    StatementKind kind() const override { return StatementKind::Block; }
    unsigned int  line_num() const override { return line; };
    void          print(std::ostream&) const override;
    bool          check_types(Checker&) override;
};

// A block that is executed only when its boolean condition is true
struct IfBlock final : public Block {
    Magnum::Pointer<Expression> condition;
    Magnum::Pointer<Block> else_or_else_if{nullptr};

    explicit IfBlock(Expression* cond)
        : Block(cond->line_num()), condition(cond) {}

    StatementKind kind() const override { return StatementKind::IfBlock; }
    void          print(std::ostream&) const override;
    bool          check_types(Checker&) override;
};

// A block that runs repeatedly until its condition is false
struct WhileLoop final : public Block {
    Magnum::Pointer<Expression> condition;

    explicit WhileLoop(Expression* cond)
        : Block(cond->line_num()), condition(cond) {}

    StatementKind kind() const override { return StatementKind::While; }
    void          print(std::ostream&) const override;
    bool          check_types(Checker&) override;
};

struct ReturnStatement final : public Statement {
    Magnum::Pointer<Expression> expression;

    explicit ReturnStatement(Expression* expr) : expression(expr) {}

    StatementKind kind() const override { return StatementKind::Return; }
    unsigned int  line_num() const override { return expression->line_num(); }
    void          print(std::ostream&) const override;
    bool          check_types(Checker&) override;
};

// A callable procedure that optionally takes inputs
struct Function {
    std::string name;
    const Type* return_type = &Type::Void;
    std::vector<Magnum::Pointer<LValue>> parameters;

    explicit Function(const std::string& n) : name(n) {}
    Function(const Function&) = default;
    Function(Function&&) = default;
    Function& operator=(const Function&) = default;
    Function& operator=(Function&&) = default;
    virtual ~Function() noexcept = default;

    virtual void         print(std::ostream&) const = 0;
    virtual FunctionKind kind() const = 0;
    virtual unsigned int line_num() const = 0;
};

// A procedure written in Bluebird containing statements and
// optionally inputs/outputs
struct BBFunction final : public Function {
    Block body;

    explicit BBFunction(const std::string& n, unsigned int line = 0)
        : Function(n), body(line) {}

    void         print(std::ostream&) const override;
    FunctionKind kind() const override { return FunctionKind::Normal; }
    unsigned int line_num() const override { return body.line_num(); }
};

// A function with no body (written in Bluebird, that is); forward
// declares some function (likely in C) of some other library/object file
struct BuiltinFunction final : public Function {
    bool is_used = false;

    explicit BuiltinFunction(const std::string& n) : Function(n) {}

    void         print(std::ostream&) const override;
    FunctionKind kind() const override { return FunctionKind::Builtin; }
    unsigned int line_num() const override { return 0; }
};
#endif
