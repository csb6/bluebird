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
#include "checker.h"
#include "ast.h"
#include "error.h"
#include "visitor.h"
#include <cassert>

class CheckerExprVisitor : public ExprVisitor<CheckerExprVisitor> {
public:
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

class CheckerStmtVisitor : public StmtVisitor<CheckerStmtVisitor> {
    BBFunction* m_curr_funct;
public:
    explicit CheckerStmtVisitor(BBFunction* curr_funct) : m_curr_funct(curr_funct) {}

    bool on_visit(BasicStatement&);
    bool on_visit(Initialization&);
    bool on_visit(Assignment&);
    bool on_visit(IfBlock&);
    bool on_visit(Block&);
    bool on_visit(WhileLoop&);
    bool on_visit(ReturnStatement&);
};

Checker::Checker(std::vector<Magnum::Pointer<Function>>& functions,
                 std::vector<Magnum::Pointer<Type>>& types,
                 std::vector<Magnum::Pointer<Initialization>>& global_vars)
    : m_functions(functions), m_types(types), m_global_vars(global_vars)
{}

template<typename Other>
[[noreturn]] static
void print_type_mismatch(const Expression* expr,
                         const Other* other, const Type* other_type,
                         const char* other_label = "Used with",
                         const char* expr_label = "Expression")
{
    Error(expr->line_num()).put("Types don't match:\n")
        .put(expr_label, 2).put(": ").put(expr).put("\t").put(expr->type()).newline()
        .put(other_label, 2).put(": ").put(other).put("\t").put(other_type).raise();
}

[[noreturn]] static
void nonbool_condition_error(const Expression* condition,
                             const char* statement_name)
{
    Error(condition->line_num()).put("Expected boolean condition for\n  ")
        .quote(statement_name).put("statement, but instead found:\n")
        .put("Expression: ", 2).put(condition).put("\t").put(condition->type()).raise();
}

// No need to typecheck within literal/variable expressions since they aren't composite

template<typename Other>
static
bool matched_ptr(Expression* expr, const Other* other, const Type* other_type)
{
    if(expr->type()->kind() != TypeKind::Ptr || other_type->kind() != TypeKind::Ptr) {
        return false;
    }
    auto* expr_ptr_type = static_cast<const PtrType*>(expr->type());
    auto* other_ptr_type = static_cast<const PtrType*>(other_type);

    if(other_ptr_type->inner_type == expr_ptr_type->inner_type
       && (other_ptr_type->is_anonymous || expr_ptr_type->is_anonymous)) {
        // Anonymous pointer types can implicitly convert to any compatible named pointer type
        return true;
    } else {
        print_type_mismatch(expr, other, other_type, "Used with", "Expression");
    }
}

static
void check_legal_unary_op(const UnaryExpression& expr, TokenType op, const Type* type)
{
    if(expr.op == TokenType::Op_To_Ptr) {
        if(expr.right->kind() != ExprKind::Variable) {
            Error(expr.line_num()).raise("Can only use to_ptr operator on named variables");
        }
        return;
    } else if(expr.op == TokenType::Op_To_Val) {
        if(type->kind() != TypeKind::Ptr) {
            Error(expr.line_num()).raise("Cannot use to_val operator on a non-pointer value");
        }
        return;
    }

    switch(type->kind()) {
    case TypeKind::IntRange:
        if(is_bool_op(op)) {
            Error(expr.right->line_num()).put("The boolean operator").quote(op)
                .put("cannot be used with expression:\n  ")
                .put(expr.right.get()).newline()
                .put("because the expression type is a range type, not a boolean type")
                .raise();
        }
        break;
    case TypeKind::Boolean:
        if(!is_bool_op(op)) {
            Error(expr.right->line_num()).put("The non-boolean operator")
                .quote(op).put("cannot be used with expression:\n ")
                .put(expr.right.get()).put("\t").put(expr.right->type()).newline()
                .raise("because the expression is a boolean type");
        }
        break;
    case TypeKind::Literal:
        // Assumes all but float literals are constant folded before this stage
        assert(false);
        break;
    default:
        Error(expr.line_num()).put("The operator").quote(op)
            .put("cannot be used with expression:\n ")
            .put(&expr).put("\t").put(expr.type()).raise();
    }
}

static
void check_legal_bin_op(const Expression& expr, TokenType op, const Type* type)
{
    switch(type->kind()) {
    case TypeKind::IntRange:
    case TypeKind::Boolean:
        break;
    case TypeKind::Literal:
        if(expr.kind() == ExprKind::InitList) {
            Error(expr.line_num())
                .raise("Operators cannot be used with initializer lists");
        }
        break;
    default:
        Error(expr.line_num()).put("The operator").quote(op)
            .put("cannot be used with expression:\n ")
            .put(&expr).put("\t").put(expr.type()).raise();
    }
}

void CheckerExprVisitor::on_visit(UnaryExpression& expr)
{
    visit(*expr.right);
    check_legal_unary_op(expr, expr.op, expr.right->type());
}

void CheckerExprVisitor::on_visit(BinaryExpression& expr)
{
    // Ensure the sub-expressions are correct first
    visit(*expr.left);
    visit(*expr.right);
    const Type* left_type = expr.left->type();
    const Type* right_type = expr.right->type();

    check_legal_bin_op(*expr.left, expr.op, left_type);
    check_legal_bin_op(*expr.right, expr.op, right_type);
    if(left_type == right_type)
        return;

    print_type_mismatch(expr.left.get(), expr.right.get(), right_type, "Right", "Left");
}

static
void typecheck_assign(Expression*, const Assignable*);

void CheckerExprVisitor::on_visit(FunctionCall& call)
{
    if(call.arguments.size() != call.definition->parameters.size()) {
        Error(call.line_num()).put("Function").quote(call.name()).put("expects ")
            .put(call.definition->parameters.size()).put(" arguments, but ")
            .put(call.arguments.size()).raise(" were provided");
    }
    const size_t arg_count = call.arguments.size();
    for(size_t i = 0; i < arg_count; ++i) {
        typecheck_assign(call.arguments[i].get(), call.definition->parameters[i].get());
    }
}

void CheckerExprVisitor::on_visit(IndexOp& expr)
{
    if(expr.base_expr->kind() != ExprKind::Variable) {
        Error(expr.line_num()).put(" Cannot index into the expression:\n  ")
            .put(expr.base_expr.get()).put("\t").put(expr.base_expr->type()).raise();
    }
    visit(*expr.base_expr);
    auto* base_type = expr.base_expr->type();
    if(base_type->kind() != TypeKind::Array) {
        Error(expr.line_num()).put(" Object\n ")
            .put(expr.base_expr.get()).put("\t").put(base_type).newline()
            .raise(" is not an array type and so cannot be indexed using `[ ]`");
    }
    auto* arr_type = static_cast<const ArrayType*>(base_type);
    visit(*expr.index_expr);
    if(expr.index_expr->type() != arr_type->index_type) {
        print_type_mismatch(expr.index_expr.get(), expr.base_expr.get(),
                            arr_type->index_type, "Expected array index", "Actual");
    }
}

void CheckerExprVisitor::on_visit(InitList& init_list)
{
    // InitLists are not typechecked like most expressions since typechecking them
    // requires knowing info. about the variable they are being assigned to.
    // typecheck_init_list() handles all typechecking in these contexts. If
    // this function gets called, then the init list is being used incorrectly.
    Error(init_list.line_num())
        .raise("Initialization lists cannot be used as part of a larger "
               "expression. They can only be used as initialization or "
               "assignment expressions.");
}

static
void typecheck_init_list(InitList* init_list, const Assignable* assignable)
{
    // Assertion should be made true in cleanup pass
    assert(assignable->type->kind() == TypeKind::Array);
    const auto* assignable_type = static_cast<const ArrayType*>(assignable->type);

    // TODO: zero-initialize all other indices
    if(init_list->values.size() > assignable_type->index_type->range.size()) {
        Error(init_list->line_num()).put(" Array ").put(assignable_type)
            .put(" expects at most ").put(assignable_type->index_type->range.size())
            .put(" values, but this initializer list provides ")
            .put(init_list->values.size()).raise(" value(s)");
    }

    for(auto& value : init_list->values) {
        CheckerExprVisitor().visit(*value);
        if(value->type() != assignable_type->element_type) {
            print_type_mismatch(value.get(), assignable, assignable_type->element_type,
                                "Expected initializer list item", "Actual item");
        }
    }
}

static
void typecheck_assign(Expression* assign_expr, const Assignable* assignable)
{
    if(assign_expr->kind() == ExprKind::InitList) {
        typecheck_init_list(static_cast<InitList*>(assign_expr), assignable);
        return;
    }

    CheckerExprVisitor().visit(*assign_expr);
    if(assign_expr->type() == assignable->type || matched_ptr(assign_expr, assignable, assignable->type)) {
        return;
    }

    print_type_mismatch(assign_expr, assignable, assignable->type);
}

bool CheckerStmtVisitor::on_visit(BasicStatement& stmt)
{
    CheckerExprVisitor().visit(*stmt.expression);
    if(stmt.expression->kind() == ExprKind::Binary) {
        auto* bin_expr = static_cast<const BinaryExpression*>(stmt.expression.get());
        if(bin_expr->op == TokenType::Op_Eq) {
            Error(bin_expr->line_num())
                .raise("Equality operator is `=`, assignment operator"
                       " is `:=`; use assignment");
        }
    }
    return false;
}

bool CheckerStmtVisitor::on_visit(Initialization& init_stmt)
{
    if(init_stmt.expression != nullptr) {
        typecheck_assign(init_stmt.expression.get(), init_stmt.variable.get());
    } else if(init_stmt.variable->type->kind() == TypeKind::Ptr) {
        Error(init_stmt.line_num())
            .raise("Pointer variables must be given an initial value");
    }
    return false;
}

bool CheckerStmtVisitor::on_visit(Assignment& assgn_stmt)
{
    typecheck_assign(assgn_stmt.expression.get(), assgn_stmt.assignable);
    if(assgn_stmt.assignable->kind() == AssignableKind::Indexed) {
        CheckerExprVisitor()
            .visit(*static_cast<IndexedVariable*>(assgn_stmt.assignable)->array_access);
    }
    return false;
}

static
bool is_bool_condition(const Expression* condition)
{
    const Type* cond_type = condition->type();
    return cond_type->kind() == TypeKind::Boolean
        || cond_type == &LiteralType::Bool;
}

bool CheckerStmtVisitor::on_visit(IfBlock& if_block)
{
    if(!is_bool_condition(if_block.condition.get())) {
        nonbool_condition_error(if_block.condition.get(), "if-statement");
    }
    CheckerExprVisitor().visit(*if_block.condition);
    bool always_returns = on_visit(static_cast<Block&>(if_block));
    if(if_block.else_or_else_if != nullptr) {
        always_returns &= visit(*if_block.else_or_else_if);
    } else {
        // Since an if-statement does not cover all possibilities
        always_returns = false;
    }
    return always_returns;
}

bool CheckerStmtVisitor::on_visit(Block& block)
{
    bool always_returns = false;
    for(auto& stmt : block.statements) {
        always_returns |= visit(*stmt);

        if(always_returns && stmt.get() != block.statements.back().get()) {
            Error(block.line_num()).put(" Statements after:\n  ").put(stmt.get())
                .raise("\n are unreachable because the statement always returns");
        }
    }
    return always_returns;
}

bool CheckerStmtVisitor::on_visit(WhileLoop& loop)
{
    CheckerExprVisitor().visit(*loop.condition);
    if(!is_bool_condition(loop.condition.get())) {
        nonbool_condition_error(loop.condition.get(), "while-loop");
    }
    on_visit(static_cast<Block&>(loop));
    return false;
}

bool CheckerStmtVisitor::on_visit(ReturnStatement& ret_stmt)
{
    assert(m_curr_funct != nullptr);
    if(ret_stmt.expression.get() == nullptr) {
        if(m_curr_funct->return_type != &Type::Void) {
            Error(ret_stmt.line_num()).put("Function").quote(m_curr_funct->name)
                .raise("returns a value, so an empty return statement isn't allowed");
        }
        return true;
    }
    CheckerExprVisitor().visit(*ret_stmt.expression);

    const Type* return_type = ret_stmt.expression->type();
    if(m_curr_funct->return_type == return_type) {
        return true;
    } else if(m_curr_funct->return_type == &Type::Void) {
        Error(ret_stmt.expression->line_num())
            .put("Did not expect void function").quote(m_curr_funct->name)
            .raise("to return something");
    } else {
        Error(ret_stmt.expression->line_num())
            .put("Expected function").quote(m_curr_funct->name)
            .put("to return:\n  ").put(m_curr_funct->return_type).newline()
            .put("not:\n  ").put(return_type).raise();
    }
}

void Checker::run()
{
    for(auto& var : m_global_vars) {
        CheckerStmtVisitor(nullptr).visit(*var);
        if(var->expression != nullptr) {
            switch(var->expression->kind()) {
            case ExprKind::CharLiteral:
            case ExprKind::IntLiteral:
                break;
            default:
                Error(var->expression->line_num())
                    .raise(" Global variables/constants can only be initialized"
                           " to integer or character literals");
            }
        }
    }

    for(auto& function : m_functions) {
        if(function->kind() == FunctionKind::Normal) {
            auto* curr_funct = static_cast<BBFunction*>(function.get());
            bool always_returns = CheckerStmtVisitor(curr_funct).visit(curr_funct->body);
            if(!always_returns && curr_funct->return_type != &Type::Void) {
                Error().put("Function").quote(curr_funct->name)
                    .raise("does not return in all cases");
            }
        }
    }
}
