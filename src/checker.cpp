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

// No need to typecheck for literal/lvalue expressions since they aren't composite,
// so each of their check_types() member functions are empty, defined in header

static
void typecheck_init_list(InitList*, const LValue*, const ArrayType*);

template<typename Other>
static
bool matched_literal(Expression* literal, const Other* other,
                     const Type* other_type)
{
    if(literal->type()->kind() != TypeKind::Literal)
        return false;

    switch(other_type->kind()) {
    case TypeKind::Range: {
        auto* range_type = static_cast<const RangeType*>(other_type);
        const IntRange& range = range_type->range;
        switch(literal->kind()) {
        case ExprKind::IntLiteral: {
            auto* int_literal = static_cast<IntLiteral*>(literal);
            if(!range.contains(int_literal->value)) {
                Error(int_literal->line_num())
                    .put("Integer Literal ").put(int_literal)
                    .put(" is not in the range of:\n  ")
                    .put(range_type).newline()
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
                    .put(other_type).raise();
            }
            char_literal->actual_type = other_type;
            break;
        }
        default:
            Error(literal->line_num())
                .put("Expected a literal type, but instead found expression:\n  ")
                .put(literal).put("\t").put(literal->type()).raise();
        }
        break;
    }
    case TypeKind::Boolean:
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
        break;
    case TypeKind::Array:
        if constexpr (std::is_same_v<decltype(other), const LValue*>) {
            if(literal->kind() == ExprKind::InitList) {
                auto* init_list = static_cast<InitList*>(literal);
                auto* lvalue_type = static_cast<const ArrayType*>(other_type);
                typecheck_init_list(init_list, other, lvalue_type);
                init_list->set(other_type);
            } else {
                Error(literal->line_num())
                    .put("Expected an initializer list, but instead found expression:\n  ")
                    .put(literal).put("\t").put(literal->type()).raise();
            }
        } else {
            print_type_mismatch(literal, other, other_type, "Used with", "Literal");
        }
        break;
    case TypeKind::Ref:
        // References can be implicitly dereferenced when used in expressions
        return matched_literal(literal, other,
                               static_cast<const RefType*>(other_type)->inner_type);
    default:
        print_type_mismatch(literal, other, other_type, "Used with", "Literal");
    }
    return true;
}

template<typename Other>
static
bool matched_ref(Expression* expr, const Other* other, const Type* other_type)
{
    if(other_type->kind() != TypeKind::Ref) {
        return false;
    }

    auto* ref_type = static_cast<const RefType*>(other_type);
    if(expr->kind() == ExprKind::LValue && ref_type->inner_type == expr->type()) {
        return true;
    } else {
        print_type_mismatch(expr, other, other_type, "Used with", "Expression");
    }
}

static
void check_legal_unary_op(const UnaryExpression* expr, TokenType op, const Type* type)
{
    switch(type->kind()) {
    case TypeKind::Range:
        if(is_bool_op(op)) {
            Error(expr->right->line_num()).put("The boolean operator").quote(op)
                .put("cannot be used with expression:\n  ")
                .put(expr->right.get()).newline()
                .put("because the expression type is a range type, not a boolean type")
                .raise();
        }
        break;
    case TypeKind::Boolean:
        if(!is_bool_op(op)) {
            Error(expr->right->line_num()).put("The non-boolean operator")
                .quote(op).put("cannot be used with expression:\n ")
                .put(expr->right.get()).put("\t").put(expr->right->type()).newline()
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
        Error(expr->line_num()).put("The operator").quote(op)
            .put("cannot be used with expression:\n ")
            .put(expr).put("\t").put(expr->type()).raise();
    }
}

static
void check_legal_bin_op(const Expression* expr, TokenType op, const Type* type)
{
    switch(type->kind()) {
    case TypeKind::Range:
    case TypeKind::Boolean:
        break;
    case TypeKind::Literal:
        if(expr->kind() == ExprKind::InitList) {
            Error(expr->line_num())
                .raise("Operators cannot be used with initializer lists");
        }
        break;
    case TypeKind::Ref:
        check_legal_bin_op(expr, op, static_cast<const RefType*>(type)->inner_type);
        break;
    default:
        Error(expr->line_num()).put("The operator").quote(op)
            .put("cannot be used with expression:\n ")
            .put(expr).put("\t").put(expr->type()).raise();
    }
}

void UnaryExpression::check_types()
{
    right->check_types();
    check_legal_unary_op(this, op, right->type());
}

// TODO: check if this binary op is legal for this type
void BinaryExpression::check_types()
{
    // Ensure the sub-expressions are correct first
    left->check_types();
    right->check_types();
    const Type* left_type = left->type();
    const Type* right_type = right->type();

    check_legal_bin_op(left.get(), op, left_type);
    check_legal_bin_op(right.get(), op, right_type);
    if(left_type == right_type
       || matched_literal(left.get(), right.get(), right_type)
       || matched_literal(right.get(), left.get(), left_type)
       || matched_ref(left.get(), right.get(), right_type)
       || matched_ref(right.get(), left.get(), left_type))
        return;

    print_type_mismatch(left.get(), right.get(), right_type, "Right", "Left");
}

static
void typecheck_assign(Magnum::Pointer<Expression>& assign_expr,
                      const LValue* assign_lval)
{
    assign_expr->check_types();
    const auto* assign_expr_type = assign_expr->type();
    if(assign_expr_type == assign_lval->type
       || matched_literal(assign_expr.get(), assign_lval, assign_lval->type)
       || matched_ref(assign_expr.get(), assign_lval, assign_lval->type))
        return;

    print_type_mismatch(assign_expr.get(), assign_lval, assign_lval->type);
}

void FunctionCall::check_types()
{
    if(arguments.size() != definition->parameters.size()) {
        Error(line_num()).put("Function").quote(name()).put("expects ")
            .put(definition->parameters.size()).put(" arguments, but ")
            .put(arguments.size()).raise(" were provided");
    }
    const size_t arg_count = arguments.size();
    for(size_t i = 0; i < arg_count; ++i) {
        typecheck_assign(arguments[i], definition->parameters[i].get());
    }
}

void IndexOp::check_types()
{
    if(base_expr->kind() != ExprKind::LValue) {
        Error(line_num()).put(" Cannot index into the expression:\n  ")
            .put(base_expr.get()).put("\t").put(base_expr->type()).raise();
    }
    base_expr->check_types();
    auto* base_type = base_expr->type();
    if(base_type->kind() != TypeKind::Array) {
        Error(line_num()).put(" Object\n ")
            .put(base_expr.get()).put("\t").put(base_type).newline()
            .raise(" is not an array type and so cannot be indexed using `[ ]`");
    }
    auto* arr_type = static_cast<const ArrayType*>(base_type);
    index_expr->check_types();
    if(index_expr->type() == arr_type->index_type
       || matched_literal(index_expr.get(), base_expr.get(), arr_type->index_type)) {
        return;
    } else {
        print_type_mismatch(index_expr.get(), base_expr.get(), arr_type->index_type,
                            "Expected array index", "Actual");
    }
}

static
void typecheck_init_list(InitList* init_list, const LValue* lvalue,
                         const ArrayType* lvalue_type)
{
    // TODO: zero-initialize all other indices
    if(init_list->values.size() > lvalue_type->index_type->range.size()) {
        Error(init_list->line_num()).put(" Array ").put(lvalue_type)
            .put(" expects at most ").put(lvalue_type->index_type->range.size())
            .put(" values, but this initializer list provides ")
            .put(init_list->values.size()).raise(" value(s)");
    }

    for(auto& value : init_list->values) {
        value->check_types();
        if(value->type() == lvalue_type->element_type
           || matched_literal(value.get(), lvalue, lvalue_type->element_type)) {
            continue;
        } else {
            print_type_mismatch(value.get(), lvalue, lvalue_type->element_type,
                                "Expected initializer list item", "Actual item");
        }
    }
}

void InitList::check_types()
{
    if(use_kind != Kind::InInit) {
        // This is an InitList literal (i.e. it is not part of an Initialization
        // statement). It will be typechecked by match_literals.
        return;
    }

    if(lvalue->type->kind() == TypeKind::Array) {
        typecheck_init_list(this, lvalue, static_cast<const ArrayType*>(lvalue->type));
    } else {
        // TODO: add support for record type initializer lists
        // Non-aggregate types cannot have initializer lists
        Error(line_num())
            .raise("Initializer lists can only be used to initialize/assign"
                   " a variable or constant of array/record types");
    }
}

bool BasicStatement::check_types(Checker&)
{
    expression->check_types();
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
    if(expression == nullptr) {
        if(lvalue->type->kind() == TypeKind::Ref) {
            Error(line_num()).raise("Reference variables must be given an initial value");
        }
    } else {
        typecheck_assign(expression, lvalue.get());
    }
    return false;
}

bool Assignment::check_types(Checker&)
{
    typecheck_assign(expression, lvalue);
    if(lvalue->kind() == LValueKind::Index) {
        static_cast<IndexLValue*>(lvalue)->array_access->check_types();
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
    condition->check_types();
    if(!is_bool_condition(condition.get())) {
        nonbool_condition_error(condition.get(), "if-statement");
    }
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
    condition->check_types();
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
    expression->check_types();

    const Type* return_type = expression->type();
    if(curr_funct->return_type == return_type
       || matched_literal(expression.get(), curr_funct, curr_funct->return_type)) {
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
