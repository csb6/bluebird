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
    void on_visit(IndexedExpr&);
    void on_visit(InitList&);

    void visit_and_fold(Magnum::Pointer<Expression>&);
};

class CleanupStmtVisitor : public StmtVisitor<CleanupStmtVisitor> {
    BBFunction* m_curr_funct;
    std::unordered_map<const Type*, Magnum::Pointer<PtrType>>& m_anon_ptr_types;
    CleanupExprVisitor m_expr_visitor;
public:
    CleanupStmtVisitor(BBFunction* curr_funct,
                       std::unordered_map<const Type*, Magnum::Pointer<PtrType>>& anon_ptr_types)
        : m_curr_funct(curr_funct), m_anon_ptr_types(anon_ptr_types), m_expr_visitor(m_anon_ptr_types) {}

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
void infer_literal_intrange_type(Expression* literal, const Other* other,
                                 const IntRangeType* other_type)
{
    switch(literal->kind()) {
    case ExprKind::IntLiteral: {
        auto* int_literal = as<IntLiteral>(literal);
        if(!other_type->range.contains(int_literal->value)) {
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
        auto* char_literal = as<CharLiteral>(literal);
        if(other_type != &IntRangeType::Character) {
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
void infer_literal_floatrange_type(Expression* literal, const FloatRangeType* other_type)
{
    if(literal->kind() == ExprKind::FloatLiteral) {
        auto* float_literal = as<FloatLiteral>(literal);
        float_literal->actual_type = other_type;
    } else {
        Error(literal->line_num())
            .put("Expected a float literal, but instead found expression:\n  ")
            .put(literal).put("\t").put(literal->type()).raise();
    }
}

static
void infer_literal_bool_type(Expression* literal, const Type* other_type)
{
    // TODO: add support for user-created boolean types, which should be usable
    // with Boolean literals
    if(literal->kind() == ExprKind::BoolLiteral) {
        as<BoolLiteral>(literal)->actual_type = other_type;
    } else {
        Error(literal->line_num())
            .put("Expected a boolean literal, but instead found expression:\n  ")
            .put(literal).put("\t").put(literal->type()).raise();
    }
}

static
void infer_literal_array_type(Expression*, const ArrayType*);

template<typename Other>
static
void infer_literal_type(Expression* literal, const Other* other, const Type* other_type)
{
    if(literal->type()->kind() != TypeKind::Literal) {
        return;
    }

    switch(other_type->kind()) {
    case TypeKind::IntRange:
        infer_literal_intrange_type(literal, other, as<IntRangeType>(other_type));
        break;
    case TypeKind::FloatRange:
        infer_literal_floatrange_type(literal, as<FloatRangeType>(other_type));
        break;
    case TypeKind::Boolean:
        infer_literal_bool_type(literal, other_type);
        break;
    case TypeKind::Array:
        infer_literal_array_type(literal, as<ArrayType>(other_type));
        break;
    case TypeKind::Ptr:
        Error(literal->line_num()).put("Literal ").put(literal).raise(" cannot be used with pointer types");
        break;
    case TypeKind::Literal:
        // Ignore case of two Literal operands; will get constant folded
        break;
    case TypeKind::Normal:
        BLUEBIRD_UNREACHABLE("Unexpected TypeKind::Normal");
        break;
    }
}

static
void infer_literal_array_type(Expression* literal, const ArrayType* other_type)
{
    if(literal->kind() == ExprKind::InitList) {
        auto* init_list = as<InitList>(literal);
        init_list->actual_type = other_type;
        for(auto& expr : init_list->values) {
            infer_literal_type(expr.get(), init_list, other_type->element_type);
        }
    } else {
        Error(literal->line_num())
            .put("Expected an initializer list, but instead found expression:\n  ")
            .put(literal).put("\t").put(literal->type()).raise();
    }
}

void CleanupExprVisitor::visit_and_fold(Magnum::Pointer<Expression>& expr)
{
    visit(*expr);
    fold_constants(expr);
}

void CleanupExprVisitor::on_visit(UnaryExpression& expr)
{
    visit_and_fold(expr.right);

    if(expr.op == TokenType::Op_To_Ptr) {
        // The to_ptr operation takes the address of right's type (T). The unary expression's type
        // then becomes T*. This is necessary to do here so that typechecking can proceed properly.
        const auto* right_type = expr.type();
        if(auto match = m_anon_ptr_types.find(right_type); match != m_anon_ptr_types.end()) {
            expr.actual_type = match->second.get();
        } else {
            auto[it, success] = m_anon_ptr_types.emplace(right_type, Magnum::pointer<PtrType>(right_type));
            assert(success);
            expr.actual_type = it->second.get();
        }
    } else if(expr.op == TokenType::Op_To_Val) {
        // The to_val operation dereferences a pointer of type T*. The unary expression's type
        // then becomes T.
        const auto* right_type = expr.type();
        if(right_type->kind() != TypeKind::Ptr) {
            Error(expr.line_num()).raise("Cannot use to_val operator on a non-pointer value");
        } else {
            expr.actual_type = as<PtrType>(right_type)->inner_type;
        }
    }
}

void CleanupExprVisitor::on_visit(BinaryExpression& expr)
{
    visit_and_fold(expr.left);
    visit_and_fold(expr.right);
    infer_literal_type(expr.left.get(), expr.right.get(), expr.right->type());
    infer_literal_type(expr.right.get(), expr.left.get(), expr.left->type());
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
        visit_and_fold(arg);
        infer_literal_type(arg.get(), param->get(), (*param)->type);
        ++param;
    }
}

void CleanupExprVisitor::on_visit(IndexedExpr& expr)
{
    // Base expression will not consist of anything that can be simplified down
    // to a single literal (because it is some sort of variable usage), so no need
    // to try and fold it; instead, just visit() it so any of its subexpressions
    // get folded if possible.
    visit(*expr.base_expr);
    visit_and_fold(expr.index_expr);

    const auto* base_type = expr.base_expr->type();
    if(base_type->kind() != TypeKind::Array) {
        Error(expr.line_num()).put(" Object\n ")
            .put(expr.base_expr.get()).put("\t").put(base_type).newline()
            .raise(" is not an array type and so cannot be indexed using `[ ]`");
    }
    infer_literal_type(expr.index_expr.get(), expr.base_expr.get(), as<ArrayType>(base_type)->index_type);
}

void CleanupExprVisitor::on_visit(InitList& expr)
{
    for(auto& val : expr.values) {
        visit_and_fold(val);
    }
}


void CleanupStmtVisitor::on_visit(BasicStatement& stmt)
{
    m_expr_visitor.visit_and_fold(stmt.expression);
}

void CleanupStmtVisitor::on_visit(Initialization& stmt)
{
    if(stmt.expression != nullptr) {
        m_expr_visitor.visit_and_fold(stmt.expression);
        infer_literal_type(stmt.expression.get(), stmt.variable.get(),
                         stmt.variable->type);
    }
}

void CleanupStmtVisitor::on_visit(Assignment& stmt)
{
    m_expr_visitor.visit_and_fold(stmt.expression);
    infer_literal_type(stmt.expression.get(), stmt.assignable, stmt.assignable->type);
    if(stmt.assignable->kind() == AssignableKind::Indexed) {
        auto* indexed_var = as<IndexedVariable>(stmt.assignable);
        m_expr_visitor.visit(*indexed_var->indexed_expr);
    }
}

void CleanupStmtVisitor::on_visit(IfBlock& stmt)
{
    m_expr_visitor.visit_and_fold(stmt.condition);
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
    m_expr_visitor.visit_and_fold(loop.condition);
    on_visit(static_cast<Block&>(loop));
}

void CleanupStmtVisitor::on_visit(ReturnStatement& stmt)
{
    assert(m_curr_funct != nullptr);
    if(stmt.expression != nullptr) {
        m_expr_visitor.visit_and_fold(stmt.expression);
        infer_literal_type(stmt.expression.get(), m_curr_funct,
                         m_curr_funct->return_type);
    }
}

void Cleanup::run()
{
    {
        CleanupStmtVisitor global_var_visitor{nullptr, m_anon_ptr_types};
        for(Magnum::Pointer<Initialization>& var : m_global_vars) {
            global_var_visitor.visit(*var);
        }
    }

    for(auto& funct : m_functions) {
        if(funct->kind() == FunctionKind::Normal) {
            auto* curr_funct = as<BBFunction>(funct.get());
            CleanupStmtVisitor(curr_funct, m_anon_ptr_types).visit(curr_funct->body);
        }
    }
}
