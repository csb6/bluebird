#ifndef AST_CLASS_H
#define AST_CLASS_H
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
#include "token.h"
#include <CorradePointer.h>
#include <vector>
#include "multiprecision.h"

namespace Magnum = Corrade::Containers;

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
    Range, Normal, Literal, Boolean, Array, Ref
};

enum class FunctionKind : char {
    Normal, Builtin
};

enum class AssignableKind : char {
    Variable, Indexed
};

// A continuous sequence of integers.
// Upper/lower bounds are inclusive.
struct IntRange {
    multi_int lower_bound, upper_bound;
    bool is_signed;
    unsigned short bit_size;

    IntRange() : is_signed(true), bit_size(0) {}
    IntRange(const multi_int& lower, const multi_int& upper);

    bool              contains(const multi_int&) const;
    // The number of integers the range contains 
    unsigned long int size() const;
};
std::ostream& operator<<(std::ostream& output, const IntRange&);

// A kind of object
struct Type {
    // Some default types that don't have to be declared
    static Type Void, String, Float;
    std::string name;

    Type() = default;
    explicit Type(const std::string &n) : name(n) {}
    virtual ~Type() noexcept = default;

    // TODO: fix bug where this is called for some boolean literals
    //  (Type::bit_size should never be called)
    virtual size_t   bit_size() const { return 1; }
    virtual TypeKind kind() const { return TypeKind::Normal; }
    virtual void     print(std::ostream&) const;
};

struct LiteralType final : public Type {
    static const LiteralType Char, Int, Bool, InitList;

    using Type::Type;

    TypeKind kind() const override { return TypeKind::Literal; }
};

// Type with limited set of named, enumerated values
struct EnumType final : public Type {
    static EnumType Boolean;
    // TODO: change category when non-bool enum types added
    TypeKind category = TypeKind::Boolean;

    using Type::Type;

    size_t   bit_size() const override { return 1; }
    TypeKind kind() const override { return category; }
};

// Type with integer bounds
struct RangeType final : public Type {
    // Some more default types that don't have to be declared
    static RangeType Integer, Character;
    IntRange range;

    using Type::Type;
    RangeType(const std::string &n,
              const multi_int& lower_limit,
              const multi_int& upper_limit)
        : Type(n), range(lower_limit, upper_limit) {}

    size_t   bit_size() const override { return range.bit_size; }
    TypeKind kind() const override { return TypeKind::Range; }
    void     print(std::ostream&) const override;
    bool     is_signed() const { return range.is_signed; }
};

struct ArrayType final : public Type {
    const RangeType* index_type;
    Type* element_type;

    using Type::Type;
    ArrayType(const std::string &n, const RangeType* ind_type, Type* el_type)
        : Type(n), index_type(ind_type), element_type(el_type) {}

    size_t   bit_size() const override;
    TypeKind kind() const override { return TypeKind::Array; }
    void     print(std::ostream&) const override;
};

// A special kind of pointer that can only point to valid objects on stack;
// is never null and can't be returned/stored in records
struct RefType final : public Type {
    Type* inner_type;

    using Type::Type;
    RefType(const std::string& n, Type* t) : Type(n), inner_type(t) {}

    size_t   bit_size() const override { return sizeof(int*); }
    TypeKind kind() const override { return TypeKind::Ref; }
    void     print(std::ostream&) const override;
};

// An abstract object or non-standalone group of expressions
struct Expression {
    virtual ~Expression() noexcept = default;

    // What kind of expression this is (e.g. a literal, variable expr, binary expr, etc.)
    virtual ExprKind     kind() const = 0;
    // What the type (in the language) this expression is. May be calculated when
    // called by visiting child nodes
    virtual const Type*  type() const = 0;
    virtual unsigned int line_num() const = 0;
    virtual void         print(std::ostream&) const = 0;

    // Used in cleanup stage between parsing/typechecking. Definitions/overrides
    // defined in cleanup.cpp
    virtual void         cleanup() = 0;
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

    ExprKind     kind() const override { return ExprKind::StringLiteral; }
    const Type*  type() const override { return &Type::String; }
    unsigned int line_num() const override { return line; }
    void         print(std::ostream&) const override;

    void         cleanup() override {}
    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct CharLiteral final : public Expression {
    char value;
    unsigned int line;
    const Type* actual_type = &LiteralType::Char;

    CharLiteral(unsigned int line_n, char v) : value(v), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::CharLiteral; }
    const Type*  type() const override { return actual_type; }
    unsigned int line_num() const override { return line; }
    void         print(std::ostream&) const override;

    void         cleanup() override {}
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

    ExprKind     kind() const override { return ExprKind::IntLiteral; }
    const Type*  type() const override { return actual_type; }
    unsigned int line_num() const override { return line; }
    void         print(std::ostream&) const override;

    void         cleanup() override {}
    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct BoolLiteral final : public Expression {
    bool value;
    unsigned int line;
    const Type* actual_type = &LiteralType::Bool;

    BoolLiteral(unsigned int line_n, bool v) : value(v), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::BoolLiteral; }
    const Type*  type() const override { return actual_type; }
    unsigned int line_num() const override { return line; }
    void         print(std::ostream&) const override;

    void         cleanup() override {}
    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

struct FloatLiteral final : public Expression {
    double value;
    unsigned int line;

    FloatLiteral(unsigned int line_n, double v) : value(v), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::FloatLiteral; }
    const Type*  type() const override { return &Type::Float; }
    unsigned int line_num() const override { return line; }
    void         print(std::ostream&) const override;

    void         cleanup() override {}
    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
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
    void         print(std::ostream&) const override;

    void         cleanup() override {}
    void         check_types() override {}
    llvm::Value* codegen(CodeGenerator&) override;
};

// An expression that consists of an operator and an expression
struct UnaryExpression final : public Expression {
    TokenType op;
    Magnum::Pointer<Expression> right;

    UnaryExpression(TokenType tok, Magnum::Pointer<Expression>&& expr)
        : op(tok), right(std::forward<decltype(right)>(expr)) {}

    ExprKind     kind() const override { return ExprKind::Unary; }
    const Type*  type() const override { return right->type(); }
    unsigned int line_num() const override { return right->line_num(); }
    void         print(std::ostream&) const override;

    void         cleanup() override;
    void         check_types() override;
    llvm::Value* codegen(CodeGenerator&) override;
};

// An expression that consists of an operator and two expressions
struct BinaryExpression final : public Expression {
    Magnum::Pointer<Expression> left;
    TokenType op;
    Magnum::Pointer<Expression> right;

    BinaryExpression(Magnum::Pointer<Expression>&& l, TokenType oper,
                     Magnum::Pointer<Expression>&& r)
        : left(std::forward<decltype(left)>(l)), op(oper),
          right(std::forward<decltype(right)>(r)) {}

    ExprKind     kind() const override { return ExprKind::Binary; }
    const Type*  type() const override;
    unsigned int line_num() const override { return left->line_num(); }
    void         print(std::ostream&) const override;

    void         cleanup() override;
    void         check_types() override;
    llvm::Value* codegen(CodeGenerator&) override;
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
    void         print(std::ostream&) const override;

    void         cleanup() override;
    void         check_types() override;
    llvm::Value* codegen(CodeGenerator&) override;
};

// An access into an array
struct IndexOp final : public Expression {
    // Evaluates into the object being indexed into 
    Magnum::Pointer<Expression> base_expr;
    // Some discrete type indexing into the base
    Magnum::Pointer<Expression> index_expr;
    unsigned int line;

    IndexOp(unsigned int line_n, Magnum::Pointer<Expression>&& b,
            Magnum::Pointer<Expression>&& ind)
        : base_expr(std::forward<decltype(base_expr)>(b)),
          index_expr(std::forward<decltype(index_expr)>(ind)), line(line_n) {}

    ExprKind     kind() const override { return ExprKind::IndexOp; }
    const Type*  type() const override;
    unsigned int line_num() const override { return line; }
    void         print(std::ostream&) const override;

    void         cleanup() override;
    void         check_types() override;
    llvm::Value* codegen(CodeGenerator&) override;
};

// A bracketed list of values assigned all at once to an array/record
struct InitList final : public Expression {
    enum class Kind : char {
        None, InInit, AnonObj
    };

    std::vector<Magnum::Pointer<Expression>> values;
    unsigned int line;
private:
    union {
        // When InitList is part of an assignment/initialization;
        // expects assignable to be set before end of parsing
        const struct Assignable *assignable;
        // When InitList is an anonymous object; takes on type of the
        // Assignable it is used with
        const Type* anon_type;
    };
    Kind use_kind = Kind::None;
public:
    InitList(unsigned int line_n) : line(line_n) {}

    void         set(const Assignable*);
    void         set(const Type*);
    ExprKind     kind() const override { return ExprKind::InitList; }
    const Type*  type() const override;
    unsigned int line_num() const override { return line; }
    void         print(std::ostream&) const override;

    void         cleanup() override;
    void         check_types() override;
    llvm::Value* codegen(CodeGenerator&) override;
};

// A location that can be assigned to
struct Assignable {
    Type* type;
    bool is_mutable = true;

    Assignable() = default;
    explicit Assignable(Type* t) : type(t) {}
    virtual ~Assignable() noexcept = default;

    virtual void           print(std::ostream&) const = 0;
    virtual AssignableKind kind() const = 0;
};

// A named object that can be assigned to (i.e. a variable or constant)
struct Variable final : public Assignable {
    std::string name;

    explicit Variable(const std::string& n) : name(n) {}
    Variable(const std::string& n, Type* t) : Assignable(t), name(n) {}

    void           print(std::ostream&) const override;
    AssignableKind kind() const override { return AssignableKind::Variable; }
};

// An element in an array that is going to be assigned to
struct IndexedVariable final : public Assignable {
    Magnum::Pointer<IndexOp> array_access;

    explicit IndexedVariable(Magnum::Pointer<IndexOp>&& op)
        : Assignable((Type*)op->type()), array_access(std::forward<decltype(array_access)>(op)) {}

    void           print(std::ostream&) const override;
    AssignableKind kind() const override { return AssignableKind::Indexed; }
};


// A standalone piece of code terminated with a semicolon and consisting
// of one or more expressions
struct Statement {
    virtual ~Statement() noexcept = default;

    virtual StmtKind     kind() const = 0;
    virtual unsigned int line_num() const = 0;
    virtual void         print(std::ostream&) const = 0;
    // Used in cleanup stage between parsing/typechecking. Definitions/overrides
    // defined in cleanup.cpp
    virtual void         cleanup(class Cleanup&) = 0;
    // Returns true if always returns, no matter code path
    virtual bool         check_types(class Checker&) = 0;
};

// A brief, usually one-line statement that holds a single expression
struct BasicStatement final : public Statement {
    Magnum::Pointer<Expression> expression;

    explicit BasicStatement(Magnum::Pointer<Expression>&& expr)
        : expression(std::forward<decltype(expression)>(expr)) {}

    StmtKind     kind() const override { return StmtKind::Basic; }
    unsigned int line_num() const override { return expression->line_num(); }
    void         print(std::ostream&) const override;
    void         cleanup(Cleanup&) override;
    bool         check_types(Checker&) override;
};

// Statement where a new variable is declared and optionally assigned the
// value of some expression
struct Initialization final : public Statement {
    Magnum::Pointer<Expression> expression{nullptr};
    Magnum::Pointer<Variable> variable;
    unsigned int line;

    explicit Initialization(unsigned int l, Magnum::Pointer<Variable>&& var)
        : variable(std::forward<decltype(variable)>(var)), line(l) {}

    StmtKind     kind() const override { return StmtKind::Initialization; }
    unsigned int line_num() const override { return line; }
    void         print(std::ostream& output) const override;
    void         cleanup(Cleanup&) override;
    bool         check_types(Checker&) override;
};

// Statement where an existing variable is given a value
struct Assignment final : public Statement {
    Magnum::Pointer<Expression> expression;
    Assignable* assignable;

    Assignment(Magnum::Pointer<Expression>&& expr, Assignable* lv)
        : expression(std::forward<decltype(expression)>(expr)), assignable(lv) {}

    StmtKind     kind() const override { return StmtKind::Assignment; }
    unsigned int line_num() const override { return expression->line_num(); }
    void         print(std::ostream& output) const override;
    void         cleanup(Cleanup&) override;
    bool         check_types(Checker&) override;
};

// A group of statements contained in a scope
struct Block : public Statement {
    std::vector<Magnum::Pointer<Statement>> statements;
    unsigned int line;

    explicit Block(unsigned int l) : line(l) {}

    StmtKind     kind() const override { return StmtKind::Block; }
    unsigned int line_num() const override { return line; };
    void         print(std::ostream&) const override;
    void         cleanup(Cleanup&) override;
    bool         check_types(Checker&) override;
};

// A block that is executed only when its boolean condition is true
struct IfBlock final : public Block {
    Magnum::Pointer<Expression> condition;
    Magnum::Pointer<Block> else_or_else_if{nullptr};

    explicit IfBlock(Magnum::Pointer<Expression>&& cond)
        : Block(cond->line_num()),
          condition(std::forward<decltype(condition)>(cond)) {}

    StmtKind kind() const override { return StmtKind::IfBlock; }
    void     print(std::ostream&) const override;
    void     cleanup(Cleanup&) override;
    bool     check_types(Checker&) override;
};

// A block that runs repeatedly until its condition is false
struct WhileLoop final : public Block {
    Magnum::Pointer<Expression> condition;

    explicit WhileLoop(Magnum::Pointer<Expression>&& cond)
        : Block(cond->line_num()),
          condition(std::forward<decltype(condition)>(cond)) {}

    StmtKind kind() const override { return StmtKind::While; }
    void     print(std::ostream&) const override;
    void     cleanup(Cleanup&) override;
    bool     check_types(Checker&) override;
};

struct ReturnStatement final : public Statement {
    Magnum::Pointer<Expression> expression;
    unsigned int line;

    explicit ReturnStatement(unsigned int line_n)
        : expression(nullptr), line(line_n) {}
    explicit ReturnStatement(Magnum::Pointer<Expression>&& expr)
        : expression(std::forward<decltype(expression)>(expr)) {}

    StmtKind     kind() const override { return StmtKind::Return; }
    unsigned int line_num() const override;
    void         print(std::ostream&) const override;
    void         cleanup(Cleanup&) override;
    bool         check_types(Checker&) override;
};

// A callable procedure that optionally takes inputs
struct Function {
    std::string name;
    Type* return_type = &Type::Void;
    std::vector<Magnum::Pointer<Variable>> parameters;

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
