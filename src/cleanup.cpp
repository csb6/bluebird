#include "cleanup.h"
#include "ast.h"
#include "visitor.h"
#include "error.h"
#include "constanteval.h"
#include <cassert>

class CleanupExprVisitor : public ExprVisitor<CleanupExprVisitor> {
public:
    void visit_impl(StringLiteral&) {}
    void visit_impl(CharLiteral&) {}
    void visit_impl(IntLiteral&) {}
    void visit_impl(BoolLiteral&) {}
    void visit_impl(FloatLiteral&) {}
    void visit_impl(VariableExpression&) {}
    void visit_impl(BinaryExpression&);
    void visit_impl(UnaryExpression&);
    void visit_impl(FunctionCall&);
    void visit_impl(IndexOp&);
    void visit_impl(InitList&);
};

class CleanupStmtVisitor : public StmtVisitor<CleanupStmtVisitor> {
    BBFunction* m_curr_funct;
public:
    explicit CleanupStmtVisitor(BBFunction* curr_funct) : m_curr_funct(curr_funct) {}

    void visit_impl(BasicStatement&);
    void visit_impl(Initialization&);
    void visit_impl(Assignment&);
    void visit_impl(IfBlock&);
    void visit_impl(Block&);
    void visit_impl(WhileLoop&);
    void visit_impl(ReturnStatement&);
};

Cleanup::Cleanup(std::vector<Magnum::Pointer<Function>>& functions,
                 std::vector<Magnum::Pointer<Initialization>>& global_vars)
    : m_functions(functions), m_global_vars(global_vars)
{}

template<typename Other>
static
void set_literal_range_type(Expression* literal, const Other* other,
                            const RangeType* other_type)
{
    const IntRange& range = other_type->range;
    switch(literal->kind()) {
    case ExprKind::IntLiteral: {
        auto* int_literal = static_cast<IntLiteral*>(literal);
        if(!range.contains(int_literal->value)) {
            Error(int_literal->line_num())
                .put("Integer Literal ").put(int_literal)
                .put(" is not in the range of:\n  ")
                .put(other_type).newline()
                .put(" so it cannot be used with:\n  ").put(other).raise();
        }
        // Literals (if used with a compatible range type) take on the type of what
        // they are used with
        int_literal->actual_type = other_type;
        break;
    }
    case ExprKind::CharLiteral: {
        auto* char_literal = static_cast<CharLiteral*>(literal);
        if(other_type != &RangeType::Character) {
            // TODO: allow user-defined character types to be used with character
            // literals
            Error(char_literal->line_num())
                .put("Character Literal ").put(char_literal)
                .put(" cannot be used with the non-character type:\n  ")
                .put(other_type).newline()
                .put("so it cannot be used with:\n  ").put(other).raise();
        }
        char_literal->actual_type = other_type;
        break;
    }
    default:
        Error(literal->line_num())
            .put("Expected a range/char literal, but instead found expression:\n  ")
            .put(literal).put("\t").put(literal->type()).raise();
    }
}

static
void set_literal_bool_type(Expression* literal, const Type* other_type)
{
    // TODO: add support for user-created boolean types, which should be usable
    // with Boolean literals
    if(literal->kind() == ExprKind::BoolLiteral) {
        auto* bool_literal = static_cast<BoolLiteral*>(literal);
        bool_literal->actual_type = other_type;
    } else {
        Error(literal->line_num())
            .put("Expected a boolean literal, but instead found expression:\n  ")
            .put(literal).put("\t").put(literal->type()).raise();
    }
}

static
void set_literal_array_type(Expression*, const ArrayType*);

template<typename Other>
static
void set_literal_type(Expression* literal, const Other* other, const Type* other_type)
{
    if(literal->type()->kind() != TypeKind::Literal)
        return;

    switch(other_type->kind()) {
    case TypeKind::Range:
        set_literal_range_type(literal, other, static_cast<const RangeType*>(other_type));
        break;
    case TypeKind::Boolean:
        set_literal_bool_type(literal, other_type);
        break;
    case TypeKind::Array:
        set_literal_array_type(literal, static_cast<const ArrayType*>(other_type));
        break;
    case TypeKind::Ref:
        // References can be implicitly dereferenced when used in expressions.
        // Since literals are always values, not references, try to resolve their
        // type to the dereferenced type of the reference
        set_literal_type(literal, other,
                         static_cast<const RefType*>(other_type)->inner_type);
        break;
    case TypeKind::Literal:
        // Ignore case of two Literal operands; will get constant folded
        break;
    case TypeKind::Normal:
        assert(false);
        break;
    }
}

static
void set_literal_array_type(Expression* literal, const ArrayType* other_type)
{
    if(literal->kind() == ExprKind::InitList) {
        auto* init_list = static_cast<InitList*>(literal);
        init_list->actual_type = other_type;
        for(auto& expr : init_list->values) {
            set_literal_type(expr.get(), init_list, other_type->element_type);
        }
    } else {
        Error(literal->line_num())
            .put("Expected an initializer list, but instead found expression:\n  ")
            .put(literal).put("\t").put(literal->type()).raise();
    }
}

static
void visit_child(Magnum::Pointer<Expression>& child)
{
    CleanupExprVisitor().visit(*child);
    fold_constants(child);
}

void CleanupExprVisitor::visit_impl(UnaryExpression& expr)
{
    visit_child(expr.right);
}

void CleanupExprVisitor::visit_impl(BinaryExpression& expr)
{
    visit_child(expr.left);
    visit_child(expr.right);
    set_literal_type(expr.left.get(), expr.right.get(), expr.right->type());
    set_literal_type(expr.right.get(), expr.left.get(), expr.left->type());
}

void CleanupExprVisitor::visit_impl(FunctionCall& call)
{
    if(call.definition->parameters.size() != call.arguments.size()) {
        Error(call.line_num()).put("Function").quote(call.name()).put("expects ")
            .put(call.definition->parameters.size()).put(" arguments, but ")
            .put(call.arguments.size()).raise(" were provided");
    }
    auto param = call.definition->parameters.begin();
    for(auto& arg : call.arguments) {
        visit_child(arg);
        set_literal_type(arg.get(), param->get(), (*param)->type);
        ++param;
    }
}

void CleanupExprVisitor::visit_impl(IndexOp& expr)
{
    // Base expression will not consist of anything that can be simplified down
    // to a single literal (because it is some sort of variable usage), so no need
    // to try and fold it; instead, just visit() it so any of its subexpressions
    // get folded if possible.
    visit(*expr.base_expr);
    visit_child(expr.index_expr);

    const auto* base_type = expr.base_expr->type();
    if(base_type->kind() != TypeKind::Array) {
        Error(expr.line_num()).put(" Object\n ")
            .put(expr.base_expr.get()).put("\t").put(base_type).newline()
            .raise(" is not an array type and so cannot be indexed using `[ ]`");
    }
    auto* array_type = static_cast<const ArrayType*>(base_type);
    set_literal_type(expr.index_expr.get(), expr.base_expr.get(), array_type->index_type);
}

void CleanupExprVisitor::visit_impl(InitList& expr)
{
    for(auto& val : expr.values) {
        visit_child(val);
    }
}


void CleanupStmtVisitor::visit_impl(BasicStatement& stmt)
{
    visit_child(stmt.expression);
}

void CleanupStmtVisitor::visit_impl(Initialization& stmt)
{
    if(stmt.expression != nullptr) {
        visit_child(stmt.expression);
        set_literal_type(stmt.expression.get(), stmt.variable.get(),
                         stmt.variable->type);
    }
}

void CleanupStmtVisitor::visit_impl(Assignment& stmt)
{
    visit_child(stmt.expression);
    set_literal_type(stmt.expression.get(), stmt.assignable, stmt.assignable->type);
    if(stmt.assignable->kind() == AssignableKind::Indexed) {
        auto* indexed_var = static_cast<IndexedVariable*>(stmt.assignable);
        CleanupExprVisitor().visit(*indexed_var->array_access);
    }
}

void CleanupStmtVisitor::visit_impl(IfBlock& stmt)
{
    visit_child(stmt.condition);
    visit_impl(static_cast<Block&>(stmt));
    if(stmt.else_or_else_if != nullptr) {
        visit(*stmt.else_or_else_if);
    }
}

void CleanupStmtVisitor::visit_impl(Block& block)
{
    for(auto& stmt : block.statements) {
        visit(*stmt);
    }
}

void CleanupStmtVisitor::visit_impl(WhileLoop& loop)
{
    visit_child(loop.condition);
    visit_impl(static_cast<Block&>(loop));
}

void CleanupStmtVisitor::visit_impl(ReturnStatement& stmt)
{
    assert(m_curr_funct != nullptr);
    if(stmt.expression != nullptr) {
        visit_child(stmt.expression);
        set_literal_type(stmt.expression.get(), m_curr_funct,
                         m_curr_funct->return_type);
    }
}

void Cleanup::run()
{
    for(Magnum::Pointer<Initialization>& var : m_global_vars) {
        CleanupStmtVisitor(nullptr).visit(*var.get());
    }
    for(auto& funct : m_functions) {
        if(funct->kind() == FunctionKind::Normal) {
            auto* curr_funct = static_cast<BBFunction*>(funct.get());
            CleanupStmtVisitor(curr_funct).visit(curr_funct->body);
        }
    }
}
