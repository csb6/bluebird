#ifndef AST_CLASS_H
#define AST_CLASS_H
/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2022  Cole Blakley

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
#include "token.h"
#include "multiprecision.h"
#include "util.h"
#include "magnum.h"
#include "error.h"
#include <vector>
#include <type_traits>

namespace llvm {
    class Value;
}

enum class NameType : char {
    Variable, Funct, DeclaredFunct, Type, DeclaredType
};

// Used in place of RTTI for differentiating between actual types of Expression*'s
enum class ExprKind : char {
    StringLiteral, CharLiteral, IntLiteral, BoolLiteral, FloatLiteral,
    Variable, Binary, Unary, FunctionCall, IndexOp, InitList
};

enum class StmtKind : char {
    Basic, Initialization, Assignment, IfBlock, Block, While, Return
};

enum class TypeKind : char {
    IntRange, FloatRange, Normal, Literal, Boolean, Array, Ptr
};

enum class FunctionKind : char {
    Normal, Builtin
};

enum class AssignableKind : char {
    Variable, Indexed, Deref
};

// A continuous sequence of integers.
// Upper/lower bounds are inclusive.
struct IntRange {
    multi_int lower_bound, upper_bound;
    bool is_signed;
    unsigned short bit_size;

    IntRange() : is_signed(true), bit_size(0) {}
    IntRange(multi_int lower, multi_int upper, bool inclusive);

    bool              contains(const multi_int&) const;
    // The number of integers the range contains
    unsigned long int size() const;
};

// A continuous range of 32-bit floating point numbers
struct FloatRange {
    float lower_bound, upper_bound;
    bool is_inclusive;

    FloatRange(float lower = 0.0f, float upper = 0.0f, bool inclusive = true)
        : lower_bound(lower), upper_bound(upper), is_inclusive(inclusive) {}

    bool contains(double) const;
    double size() const { return upper_bound - lower_bound; }
};


template<typename Subclass, typename Base>
inline
Subclass* as(Base* base)
{
    static_assert(std::is_base_of_v<Base, Subclass>);
    return static_cast<Subclass*>(base);
}

template<typename Subclass, typename Base>
inline
const Subclass* as(const Base* base)
{
    static_assert(std::is_base_of_v<Base, Subclass>);
    return static_cast<const Subclass*>(base);
}

template<typename Subclass, typename Base>
inline
Subclass& as(Base& base)
{
    static_assert(std::is_base_of_v<Base, Subclass>);
    return static_cast<Subclass&>(base);
}

template<typename Subclass, typename Base>
inline
const Subclass& as(const Base& base)
{
    static_assert(std::is_base_of_v<Base, Subclass>);
    return static_cast<const Subclass&>(base);
}

// A kind of object
struct Type {
    BLUEBIRD_COPYABLE(Type);
    BLUEBIRD_MOVEABLE(Type);
    // Some default types that don't have to be declared
    static const Type Void, String;
    std::string name;

    Type() = default;
    explicit
    Type(std::string_view n) : name(n) {}
    virtual ~Type() noexcept = default;

    virtual size_t   bit_size() const { BLUEBIRD_UNREACHABLE("Called Type::bit_size()"); }
    virtual TypeKind kind() const { return TypeKind::Normal; }
};

struct LiteralType final : public Type {
    static const LiteralType Char, Int, Float, Bool, InitList;

    using Type::Type;

    TypeKind kind() const override { return TypeKind::Literal; }
};

// Type with limited set of named, enumerated values
struct EnumType final : public Type {
    static const EnumType Boolean;
    // TODO: change category when non-bool enum types added
    TypeKind category = TypeKind::Boolean;

    using Type::Type;

    size_t   bit_size() const override { return 1; }
    TypeKind kind() const override { return category; }
};

// Type with integer bounds
struct IntRangeType final : public Type {
    // Some more default types that don't have to be declared
    static const IntRangeType Integer, Character;
    IntRange range;

    using Type::Type;
    IntRangeType(std::string_view n, IntRange range)
        : Type(n), range(std::move(range)) {}

    size_t   bit_size() const override { return range.bit_size; }
    TypeKind kind() const override { return TypeKind::IntRange; }
    bool     is_signed() const { return range.is_signed; }
};

struct FloatRangeType : public Type {
    static const FloatRangeType Float;
    FloatRange range;

    using Type::Type;
    FloatRangeType(std::string_view n, FloatRange range)
        : Type(n), range(range) {}

    TypeKind kind() const override { return TypeKind::FloatRange; }
};

struct ArrayType final : public Type {
    const IntRangeType* index_type;
    const Type* element_type;

    using Type::Type;
    ArrayType(std::string_view n, const IntRangeType* ind_type, const Type* el_type)
        : Type(n), index_type(ind_type), element_type(el_type) {}

    size_t   bit_size() const override;
    TypeKind kind() const override { return TypeKind::Array; }
};

// A pointer; is never null, but can be stored in records, arrays, etc.
struct PtrType final : public Type {
    const Type* inner_type;
    bool is_anonymous = true;

    using Type::Type;
    explicit
    PtrType(const Type* inner_type)
        : Type("ptr " + inner_type->name), inner_type(inner_type) {}
    PtrType(const std::string& name, const Type* inner_type)
        : Type(name), inner_type(inner_type) {}

    size_t bit_size() const override { return sizeof(int*); }
    TypeKind kind() const override { return TypeKind::Ptr; }
};

// An abstract object or non-standalone group of expressions
struct Expression {
    BLUEBIRD_COPYABLE(Expression);
    BLUEBIRD_MOVEABLE(Expression);
    Expression() = default;
    virtual ~Expression() noexcept = default;

    // What kind of expression this is (e.g. a literal, variable expr, binary expr, etc.)
    virtual ExprKind     kind() const = 0;
    // What the type (in the language) this expression is. May be calculated when
    // called by visiting child nodes
    virtual const Type*  type() const = 0;
    virtual unsigned int line_num() const = 0;
};

// Each type of literal is a nameless instance of data
struct StringLiteral final : public Expression {
    std::string value;
    unsigned int line;

    StringLiteral(unsigned int line_n, std::string_view value)
        : value(value), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::StringLiteral; }
    const Type*  type() const override { return &Type::String; }
    unsigned int line_num() const override { return line; }
};

struct CharLiteral final : public Expression {
    char value;
    unsigned int line;
    const Type* actual_type = &LiteralType::Char;

    CharLiteral(unsigned int line_n, char value) : value(value), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::CharLiteral; }
    const Type*  type() const override { return actual_type; }
    unsigned int line_num() const override { return line; }
};

struct IntLiteral final : public Expression {
    // Holds arbitrarily-sized integers
    multi_int value;
    unsigned int line;
    const Type* actual_type = &LiteralType::Int;

    IntLiteral(unsigned int line_n, std::string_view value)
        : value(value), line(line_n) {}
    IntLiteral(unsigned int line_n, multi_int value)
        : value(std::move(value)), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::IntLiteral; }
    const Type*  type() const override { return actual_type; }
    unsigned int line_num() const override { return line; }
};

struct BoolLiteral final : public Expression {
    bool value;
    unsigned int line;
    const Type* actual_type = &LiteralType::Bool;

    BoolLiteral(unsigned int line_n, bool v) : value(v), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::BoolLiteral; }
    const Type*  type() const override { return actual_type; }
    unsigned int line_num() const override { return line; }
};

struct FloatLiteral final : public Expression {
    float value;
    unsigned int line;
    const Type* actual_type = &LiteralType::Float;

    FloatLiteral(unsigned int line_n, float v) : value(v), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::FloatLiteral; }
    const Type*  type() const override { return actual_type; }
    unsigned int line_num() const override { return line; }
};

// An expression consisting solely of a variable's name
struct VariableExpression final : public Expression {
    const struct Variable *variable;
    unsigned int line;

    VariableExpression(unsigned int line_n, const Variable *v) : variable(v), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::Variable; }
    const Type*  type() const override;
    unsigned int line_num() const override { return line; }
    // Other data should be looked up in the corresponding Variable object
};

// An expression that consists of an operator and an expression
struct UnaryExpression final : public Expression {
    // When nullptr, type of right is type() of this expression
    const Type* actual_type = nullptr;
    Magnum::Pointer<Expression> right;
    TokenType op;

    UnaryExpression(TokenType tok, Magnum::Pointer<Expression> expr)
        : right(std::move(expr)), op(tok) {}

    ExprKind     kind() const override { return ExprKind::Unary; }
    const Type*  type() const override;
    unsigned int line_num() const override { return right->line_num(); }
};

// An expression that consists of an operator and two expressions
struct BinaryExpression final : public Expression {
    Magnum::Pointer<Expression> left;
    TokenType op;
    Magnum::Pointer<Expression> right;

    BinaryExpression(Magnum::Pointer<Expression> left, TokenType oper,
                     Magnum::Pointer<Expression> right)
        : left(std::move(left)), op(oper), right(std::move(right)) {}

    ExprKind     kind() const override { return ExprKind::Binary; }
    const Type*  type() const override;
    unsigned int line_num() const override { return left->line_num(); }
};

// A usage of a function
struct FunctionCall final : public Expression {
    std::vector<Magnum::Pointer<Expression>> arguments;
    unsigned int line;
    struct Function* definition;

    FunctionCall(unsigned int line_n, Function* def) : line(line_n), definition(def) {}

    const std::string& name() const;

    ExprKind     kind() const override { return ExprKind::FunctionCall; }
    const Type*  type() const override;
    unsigned int line_num() const override { return line; }
};

// An access into an array
struct IndexedExpr final : public Expression {
    // Evaluates into the object being indexed into 
    Magnum::Pointer<Expression> base_expr;
    // Some discrete type indexing into the base
    Magnum::Pointer<Expression> index_expr;
    unsigned int line;

    IndexedExpr(unsigned int line_n, Magnum::Pointer<Expression> b,
            Magnum::Pointer<Expression> index_expr)
        : base_expr(std::move(b)), index_expr(std::move(index_expr)), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::IndexOp; }
    const Type*  type() const override;
    unsigned int line_num() const override { return line; }
};

// A bracketed list of values assigned all at once to an array/record
struct InitList final : public Expression {
    std::vector<Magnum::Pointer<Expression>> values;
    const Type* actual_type = &LiteralType::InitList;
    unsigned int line;

    explicit
    InitList(unsigned int line_n) : line(line_n) {}

    ExprKind     kind() const override { return ExprKind::InitList; }
    const Type*  type() const override { return actual_type; }
    unsigned int line_num() const override { return line; }
};

// A location that can be assigned to
struct Assignable {
    BLUEBIRD_COPYABLE(Assignable);
    BLUEBIRD_MOVEABLE(Assignable);
    const Type* type;
    bool is_mutable = true;

    explicit
    Assignable(const Type* t) : type(t) {}
    virtual ~Assignable() noexcept = default;

    virtual AssignableKind kind() const = 0;
};

// A named object that can be assigned to (i.e. a variable or constant)
struct Variable final : public Assignable {
    std::string name;

    explicit
    Variable(std::string name) : Assignable(nullptr), name(std::move(name)) {}
    Variable(std::string name, const Type* t) : Assignable(t), name(std::move(name)) {}

    AssignableKind kind() const override { return AssignableKind::Variable; }
};

// An element in an array that is going to be assigned to
struct IndexedVariable final : public Assignable {
    Magnum::Pointer<IndexedExpr> indexed_expr;

    explicit
    IndexedVariable(Magnum::Pointer<IndexedExpr> op)
        : Assignable(op->type()), indexed_expr(std::move(op)) {}

    AssignableKind kind() const override { return AssignableKind::Indexed; }
};

// A pointer that is being dereferenced then assigned to
struct DerefLValue final : public Assignable {
    Variable& ptr_var;

    explicit
    DerefLValue(Variable& ptr_var) : Assignable(nullptr), ptr_var(ptr_var) {}

    AssignableKind kind() const override { return AssignableKind::Deref; }
};


// A standalone piece of code terminated with a semicolon and consisting
// of one or more expressions
struct Statement {
    BLUEBIRD_COPYABLE(Statement);
    BLUEBIRD_MOVEABLE(Statement);
    Statement() = default;
    virtual ~Statement() noexcept = default;

    virtual StmtKind     kind() const = 0;
    virtual unsigned int line_num() const = 0;
};

// A brief, usually one-line statement that holds a single expression
struct BasicStatement final : public Statement {
    Magnum::Pointer<Expression> expression;

    explicit
    BasicStatement(Magnum::Pointer<Expression> expr)
        : expression(std::move(expr)) {}

    StmtKind     kind() const override { return StmtKind::Basic; }
    unsigned int line_num() const override { return expression->line_num(); }
};

// Statement where a new variable is declared and optionally assigned the
// value of some expression
struct Initialization final : public Statement {
    Magnum::Pointer<Expression> expression{nullptr};
    Magnum::Pointer<Variable> variable;
    unsigned int line;

    explicit
    Initialization(unsigned int line, Magnum::Pointer<Variable> var)
        : variable(std::move(var)), line(line) {}

    StmtKind     kind() const override { return StmtKind::Initialization; }
    unsigned int line_num() const override { return line; }
};

// Statement where an existing variable is given a value
struct Assignment final : public Statement {
    Magnum::Pointer<Expression> expression;
    Assignable* assignable;

    Assignment(Magnum::Pointer<Expression> expr, Assignable* assignable)
        : expression(std::move(expr)), assignable(assignable) {}

    StmtKind     kind() const override { return StmtKind::Assignment; }
    unsigned int line_num() const override { return expression->line_num(); }
};

// A group of statements contained in a scope
struct Block : public Statement {
    std::vector<Magnum::Pointer<Statement>> statements;
    unsigned int line;

    explicit
    Block(unsigned int line) : line(line) {}

    StmtKind     kind() const override { return StmtKind::Block; }
    unsigned int line_num() const override { return line; }
};

// A block that is executed only when its boolean condition is true
struct IfBlock final : public Block {
    Magnum::Pointer<Expression> condition;
    Magnum::Pointer<Block> else_or_else_if{nullptr};

    explicit
    IfBlock(Magnum::Pointer<Expression>&& cond)
        : Block(cond->line_num()), condition(std::move(cond)) {}

    StmtKind kind() const override { return StmtKind::IfBlock; }
};

// A block that runs repeatedly until its condition is false
struct WhileLoop final : public Block {
    Magnum::Pointer<Expression> condition;

    explicit
    WhileLoop(Magnum::Pointer<Expression>&& cond)
        : Block(cond->line_num()),
          condition(std::move(cond)) {}

    StmtKind kind() const override { return StmtKind::While; }
};

struct ReturnStatement final : public Statement {
    Magnum::Pointer<Expression> expression;
    unsigned int line;

    explicit
    ReturnStatement(unsigned int line_n)
        : expression(nullptr), line(line_n) {}
    explicit
    ReturnStatement(Magnum::Pointer<Expression>&& expr)
        : expression(std::move(expr)), line(0) {}

    StmtKind     kind() const override { return StmtKind::Return; }
    unsigned int line_num() const override;
};

// A callable procedure that optionally takes inputs
struct Function {
    BLUEBIRD_COPYABLE(Function);
    BLUEBIRD_MOVEABLE(Function);

    std::string name;
    const Type* return_type = &Type::Void;
    std::vector<Magnum::Pointer<Variable>> parameters;

    explicit
    Function(std::string n) : name(std::move(n)) {}
    virtual ~Function() noexcept = default;

    virtual FunctionKind kind() const = 0;
    virtual unsigned int line_num() const = 0;
};

// A procedure written in Bluebird containing statements and
// optionally inputs/outputs
struct BBFunction final : public Function {
    Block body;

    explicit
    BBFunction(std::string n, unsigned int line = 0)
        : Function(std::move(n)), body(line) {}

    FunctionKind kind() const override { return FunctionKind::Normal; }
    unsigned int line_num() const override { return body.line_num(); }
};

// A function with no body (written in Bluebird, that is); forward
// declares some function (likely in C) of some other library/object file
struct BuiltinFunction final : public Function {
    bool is_used = false;

    explicit
    BuiltinFunction(std::string n) : Function(std::move(n)) {}

    FunctionKind kind() const override { return FunctionKind::Builtin; }
    unsigned int line_num() const override { return 0; }
};
#endif
