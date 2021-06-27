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
#include "checker.h"
#include "ast.h"
#include "error.h"
#include "visitor.h"
#include <cassert>

class CheckerExprVisitor : public ExprVisitor<CheckerExprVisitor> {
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

class CheckerStmtVisitor : public ExprVisitor<CheckerStmtVisitor> {
    BBFunction* m_curr_funct;
public:
    explicit CheckerStmtVisitor(BBFunction* curr_funct) : m_curr_funct(curr_funct) {}

    void visit_impl(BasicStatement&);
    void visit_impl(Initialization&);
    void visit_impl(Assignment&);
    void visit_impl(IfBlock&);
    void visit_impl(Block&);
    void visit_impl(WhileLoop&);
    void visit_impl(ReturnStatement&);
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

// No need to typecheck for literal/variable expressions since they aren't composite,
// so each of their check_types() member functions are empty, defined in header

template<typename Other>
static
bool matched_ref(Expression* expr, const Other* other, const Type* other_type)
{
    if(other_type->kind() != TypeKind::Ref) {
        return false;
    }

    auto* ref_type = static_cast<const RefType*>(other_type);
    if(expr->kind() == ExprKind::Variable && ref_type->inner_type == expr->type()) {
        return true;
    } else {
        print_type_mismatch(expr, other, other_type, "Used with", "Expression");
    }
}

static
void check_legal_unary_op(const UnaryExpression& expr, TokenType op, const Type* type)
{
    switch(type->kind()) {
    case TypeKind::Range:
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
    case TypeKind::Ref:
        check_legal_unary_op(expr, op, static_cast<const RefType*>(type)->inner_type);
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
    case TypeKind::Range:
    case TypeKind::Boolean:
        break;
    case TypeKind::Literal:
        if(expr.kind() == ExprKind::InitList) {
            Error(expr.line_num())
                .raise("Operators cannot be used with initializer lists");
        }
        break;
    case TypeKind::Ref:
        check_legal_bin_op(expr, op, static_cast<const RefType*>(type)->inner_type);
        break;
    default:
        Error(expr.line_num()).put("The operator").quote(op)
            .put("cannot be used with expression:\n ")
            .put(&expr).put("\t").put(expr.type()).raise();
    }
}

void CheckerExprVisitor::visit_impl(UnaryExpression& expr)
{
    visit(*expr.right);
    check_legal_unary_op(expr, expr.op, expr.right->type());
}

void CheckerExprVisitor::visit_impl(BinaryExpression& expr)
{
    // Ensure the sub-expressions are correct first
    visit(*expr.left);
    visit(*expr.right);
    const Type* left_type = expr.left->type();
    const Type* right_type = expr.right->type();

    check_legal_bin_op(*expr.left, expr.op, left_type);
    check_legal_bin_op(*expr.right, expr.op, right_type);
    if(left_type == right_type
       || matched_ref(expr.left.get(), expr.right.get(), right_type)
       || matched_ref(expr.right.get(), expr.left.get(), left_type))
        return;

    print_type_mismatch(expr.left.get(), expr.right.get(), right_type, "Right", "Left");
}

static
void typecheck_assign(Expression*, const Assignable*);

void CheckerExprVisitor::visit_impl(FunctionCall& call)
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

void CheckerExprVisitor::visit_impl(IndexOp& expr)
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

void CheckerExprVisitor::visit_impl(InitList& init_list)
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
    if(assign_expr->type() == assignable->type
       || matched_ref(assign_expr, assignable, assignable->type))
        return;

    print_type_mismatch(assign_expr, assignable, assignable->type);
}

bool BasicStatement::check_types(Checker&)
{
    CheckerExprVisitor().visit(*expression);
    if(expression->kind() == ExprKind::Binary) {
        auto* bin_expr = static_cast<const BinaryExpression*>(expression.get());
        if(bin_expr->op == TokenType::Op_Eq) {
            Error(bin_expr->line_num())
                .raise("Equality operator is `=`, assignment operator"
                       " is `:=`; use assignment");
        }
    }
    return false;
}

bool Initialization::check_types(Checker&)
{
    if(expression != nullptr) {
        typecheck_assign(expression.get(), variable.get());
    } else if(variable->type->kind() == TypeKind::Ref) {
        Error(line_num()).raise("Reference variables must be given an initial value");
    }
    return false;
}

bool Assignment::check_types(Checker&)
{
    typecheck_assign(expression.get(), assignable);
    if(assignable->kind() == AssignableKind::Indexed) {
        CheckerExprVisitor().visit(*static_cast<IndexedVariable*>(assignable)->array_access);
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

bool IfBlock::check_types(Checker& checker)
{
    if(!is_bool_condition(condition.get())) {
        nonbool_condition_error(condition.get(), "if-statement");
    }
    CheckerExprVisitor().visit(*condition);
    bool always_returns = Block::check_types(checker);
    if(else_or_else_if != nullptr) {
        always_returns &= else_or_else_if->check_types(checker);
    } else {
        // Since an if-statement does not cover all possibilities
        always_returns = false;
    }
    return always_returns;
}

bool Block::check_types(Checker& checker)
{
    bool always_returns = false;
    for(auto& stmt : statements) {
        always_returns |= stmt->check_types(checker);

        if(always_returns && stmt.get() != statements.back().get()) {
            Error(line_num()).put(" Statements after:\n  ").put(stmt.get())
                .raise("\n are unreachable because the statement always returns");
        }
    }
    return always_returns;
}

bool WhileLoop::check_types(Checker& checker)
{
    CheckerExprVisitor().visit(*condition);
    if(!is_bool_condition(condition.get())) {
        nonbool_condition_error(condition.get(), "while-loop");
    }
    Block::check_types(checker);
    return false;
}

bool ReturnStatement::check_types(Checker& checker)
{
    const BBFunction* curr_funct = checker.m_curr_funct;
    if(expression.get() == nullptr) {
        if(curr_funct->return_type != &Type::Void) {
            Error(line_num()).put("Function").quote(curr_funct->name)
                .raise("returns a value, so an empty return statement isn't allowed");
        }
        return true;
    }
    CheckerExprVisitor().visit(*expression);

    const Type* return_type = expression->type();
    if(curr_funct->return_type == return_type) {
        return true;
    } else if(curr_funct->return_type == &Type::Void) {
        Error(expression->line_num())
            .put("Did not expect void function").quote(curr_funct->name)
            .raise("to return something");
    } else {
        Error(expression->line_num())
            .put("Expected function").quote(curr_funct->name)
            .put("to return:\n  ").put(curr_funct->return_type).newline()
            .put("not:\n  ").put(return_type).raise();
    }
}

void Checker::run()
{
    for(auto& var : m_global_vars) {
        var->check_types(*this);
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
            m_curr_funct = static_cast<BBFunction*>(function.get());
            bool always_returns = m_curr_funct->body.check_types(*this);
            if(m_curr_funct->return_type->kind() == TypeKind::Ref) {
                Error().put("Function").quote(m_curr_funct->name)
                    .raise("returns a ref type, which is not allowed");
            }
            if(!always_returns && m_curr_funct->return_type != &Type::Void) {
                Error().put("Function").quote(m_curr_funct->name)
                    .raise("does not return in all cases");
            }
        }
    }
}
