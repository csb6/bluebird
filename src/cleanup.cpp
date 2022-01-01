#include "cleanup.h"
#include "ast.h"
#include "visitor.h"
#include "error.h"
#include "constanteval.h"
#include <cassert>

class CleanupExprVisitor : public ExprVisitor<CleanupExprVisitor> {
    std::unordered_map<const Type*, Magnum::Pointer<PtrType>>& m_anon_ptr_types;
public:
    CleanupExprVisitor(std::unordered_map<const Type*, Magnum::Pointer<PtrType>>& anon_ptr_types)
        : m_anon_ptr_types(anon_ptr_types) {}

    void on_visit(StringLiteral&) {}
    void on_visit(CharLiteral&) {}
    void on_visit(IntLiteral&) {}
    void on_visit(BoolLiteral&) {}
    void on_visit(FloatLiteral&) {}
    void on_visit(VariableExpression&) {}
    void on_visit(BinaryExpression&);
    void on_visit(UnaryExpression&);
    void on_visit(FunctionCall&);
    void on_visit(IndexOp&);
    void on_visit(InitList&);
};

class CleanupStmtVisitor : public StmtVisitor<CleanupStmtVisitor> {
    BBFunction* m_curr_funct;
    std::unordered_map<const Type*, Magnum::Pointer<PtrType>>& m_anon_ptr_types;
public:
    CleanupStmtVisitor(BBFunction* curr_funct,
                       std::unordered_map<const Type*, Magnum::Pointer<PtrType>>& anon_ptr_types)
        : m_curr_funct(curr_funct), m_anon_ptr_types(anon_ptr_types) {}

    void on_visit(BasicStatement&);
    void on_visit(Initialization&);
    void on_visit(Assignment&);
    void on_visit(IfBlock&);
    void on_visit(Block&);
    void on_visit(WhileLoop&);
    void on_visit(ReturnStatement&);
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
    case TypeKind::Ptr:
        Error(literal->line_num()).put("Literal ").put(literal).raise(" cannot be used with pointer types");
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
void visit_child(std::unordered_map<const Type*, Magnum::Pointer<PtrType>>& anon_ptr_types,
                 Magnum::Pointer<Expression>& child)
{
    CleanupExprVisitor(anon_ptr_types).visit(*child);
    fold_constants(child);
}

void CleanupExprVisitor::on_visit(UnaryExpression& expr)
{
    visit_child(m_anon_ptr_types, expr.right);

    if(expr.op == TokenType::Op_To_Ptr) {
        // The to_ptr operation takes the address of right's type (T). The unary expression's type
        // then becomes T*. This is necessary to do here so that typechecking can proceed properly.
        auto* right_type = expr.type();
        if(auto match = m_anon_ptr_types.find(right_type); match != m_anon_ptr_types.end()) {
            expr.actual_type = match->second.get();
        } else {
            auto[it, success] = m_anon_ptr_types.insert({right_type, Magnum::pointer<PtrType>(right_type)});
            assert(success);
            expr.actual_type = it->second.get();
        }
    } else if(expr.op == TokenType::Op_To_Val) {
        // The to_val operation dereferences a pointer of type T*. The unary expression's type
        // then becomes T.
        auto* right_type = expr.type();
        if(right_type->kind() != TypeKind::Ptr) {
            Error(expr.line_num()).raise("Cannot use to_val operator on a non-pointer value");
        } else {
            expr.actual_type = static_cast<const PtrType*>(right_type)->inner_type;
        }
    }
}

void CleanupExprVisitor::on_visit(BinaryExpression& expr)
{
    visit_child(m_anon_ptr_types, expr.left);
    visit_child(m_anon_ptr_types, expr.right);
    set_literal_type(expr.left.get(), expr.right.get(), expr.right->type());
    set_literal_type(expr.right.get(), expr.left.get(), expr.left->type());
}

void CleanupExprVisitor::on_visit(FunctionCall& call)
{
    if(call.definition->parameters.size() != call.arguments.size()) {
        Error(call.line_num()).put("Function").quote(call.name()).put("expects ")
            .put(call.definition->parameters.size()).put(" arguments, but ")
            .put(call.arguments.size()).raise(" were provided");
    }
    auto param = call.definition->parameters.begin();
    for(auto& arg : call.arguments) {
        visit_child(m_anon_ptr_types, arg);
        set_literal_type(arg.get(), param->get(), (*param)->type);
        ++param;
    }
}

void CleanupExprVisitor::on_visit(IndexOp& expr)
{
    // Base expression will not consist of anything that can be simplified down
    // to a single literal (because it is some sort of variable usage), so no need
    // to try and fold it; instead, just visit() it so any of its subexpressions
    // get folded if possible.
    visit(*expr.base_expr);
    visit_child(m_anon_ptr_types, expr.index_expr);

    const auto* base_type = expr.base_expr->type();
    if(base_type->kind() != TypeKind::Array) {
        Error(expr.line_num()).put(" Object\n ")
            .put(expr.base_expr.get()).put("\t").put(base_type).newline()
            .raise(" is not an array type and so cannot be indexed using `[ ]`");
    }
    auto* array_type = static_cast<const ArrayType*>(base_type);
    set_literal_type(expr.index_expr.get(), expr.base_expr.get(), array_type->index_type);
}

void CleanupExprVisitor::on_visit(InitList& expr)
{
    for(auto& val : expr.values) {
        visit_child(m_anon_ptr_types, val);
    }
}


void CleanupStmtVisitor::on_visit(BasicStatement& stmt)
{
    visit_child(m_anon_ptr_types, stmt.expression);
}

void CleanupStmtVisitor::on_visit(Initialization& stmt)
{
    if(stmt.expression != nullptr) {
        visit_child(m_anon_ptr_types, stmt.expression);
        set_literal_type(stmt.expression.get(), stmt.variable.get(),
                         stmt.variable->type);
    }
}

void CleanupStmtVisitor::on_visit(Assignment& stmt)
{
    visit_child(m_anon_ptr_types, stmt.expression);
    set_literal_type(stmt.expression.get(), stmt.assignable, stmt.assignable->type);
    if(stmt.assignable->kind() == AssignableKind::Indexed) {
        auto* indexed_var = static_cast<IndexedVariable*>(stmt.assignable);
        CleanupExprVisitor(m_anon_ptr_types).visit(*indexed_var->array_access);
    }
}

void CleanupStmtVisitor::on_visit(IfBlock& stmt)
{
    visit_child(m_anon_ptr_types, stmt.condition);
    on_visit(static_cast<Block&>(stmt));
    if(stmt.else_or_else_if != nullptr) {
        visit(*stmt.else_or_else_if);
    }
}

void CleanupStmtVisitor::on_visit(Block& block)
{
    for(auto& stmt : block.statements) {
        visit(*stmt);
    }
}

void CleanupStmtVisitor::on_visit(WhileLoop& loop)
{
    visit_child(m_anon_ptr_types, loop.condition);
    on_visit(static_cast<Block&>(loop));
}

void CleanupStmtVisitor::on_visit(ReturnStatement& stmt)
{
    assert(m_curr_funct != nullptr);
    if(stmt.expression != nullptr) {
        visit_child(m_anon_ptr_types, stmt.expression);
        set_literal_type(stmt.expression.get(), m_curr_funct,
                         m_curr_funct->return_type);
    }
}

void Cleanup::run()
{
    {
        CleanupStmtVisitor global_var_visitor{nullptr, m_anon_ptr_types};
        for(Magnum::Pointer<Initialization>& var : m_global_vars) {
            global_var_visitor.visit(*var.get());
        }
    }

    for(auto& funct : m_functions) {
        if(funct->kind() == FunctionKind::Normal) {
            auto* curr_funct = static_cast<BBFunction*>(funct.get());
            CleanupStmtVisitor(curr_funct, m_anon_ptr_types).visit(curr_funct->body);
        }
    }
}
