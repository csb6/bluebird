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

template<typename Other>
static void check_literal_types(Expression* literal, const Other* other,
                                const Type* other_type)
{
    if(other_type->kind() == TypeKind::Range) {
        auto* range_type = static_cast<const RangeType*>(other_type);
        const Range& range = range_type->range;
        switch(literal->kind()) {
        case ExpressionKind::IntLiteral: {
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
        case ExpressionKind::CharLiteral: {
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
    } else if(other_type->kind() == TypeKind::Boolean) {
        // TODO: add support for user-created boolean types, which should be usable
        // with Boolean literals
        if(literal->kind() == ExpressionKind::BoolLiteral) {
            auto* bool_literal = static_cast<BoolLiteral*>(literal);
            bool_literal->actual_type = other_type;
        } else {
            Error(literal->line_num())
                .put("Expected a boolean literal, but instead found expression:\n  ")
                .put(literal).put("\t").put(literal->type()).raise();
        }
    } else {
        print_type_mismatch(literal, other, other_type, "Used with", "Literal");
    }
}

// TODO: check that unary operators are valid for their values
void UnaryExpression::check_types()
{
    right->check_types();
    switch(right->type()->kind()) {
    case TypeKind::Literal:
        // Assumes all but float literals are constant folded before this stage
        assert(false);
        break;
    case TypeKind::Boolean:
        if(!is_bool_op(op)) {
            Error(right->line_num()).put("The non-boolean operator")
                .quote(op).put("cannot be used with expression:\n ")
                .put(right.get()).put("\t").put(right->type()).newline()
                .raise("because the expression is a boolean type");
        }
        break;
    case TypeKind::Range:
        if(is_bool_op(op)) {
            Error(right->line_num()).put("The boolean operator").quote(op)
                .put("cannot be used with expression:\n  ")
                .put(right.get()).newline()
                .put("because the expression type is a range type, not a boolean type")
                .raise();
        }
        break;
    default:
        Error(right->line_num()).put("The unary operator").quote(op)
            .put("cannot be used with expression:\n  ")
            .put(right.get()).put("\t").put(right->type()).raise();
    }
}

// TODO: check if this binary op is legal for this type
void BinaryExpression::check_types()
{
    // Ensure the sub-expressions are correct first
    left->check_types();
    right->check_types();
    const Type* left_type = left->type();
    const Type* right_type = right->type();
    // Check that the types of both sides of the operator match
    if(left_type != right_type) {
        if(right_type->kind() == TypeKind::Literal) {
            check_literal_types(right.get(), left.get(), left_type);
        } else if(left_type->kind() == TypeKind::Literal) {
            check_literal_types(left.get(), right.get(), right_type);
        } else {
            print_type_mismatch(left.get(), right.get(), right_type, "Right", "Left");
        }
    }
}

void FunctionCall::check_types()
{
    if(arguments.size() != function->parameters.size()) {
        Error(line_num()).put("Function").quote(name).put("expects ")
            .put(function->parameters.size()).put(" arguments, but ")
            .put(arguments.size()).raise(" were provided");
    }
    const size_t arg_count = arguments.size();
    for(size_t i = 0; i < arg_count; ++i) {
        Expression* arg = arguments[i].get();
        // Ensure each argument expression is internally typed correctly
        arg->check_types();
        // Make sure each arg type matches corresponding parameter type
        const LValue* param = function->parameters[i].get();
        if(arg->type() != param->type) {
            if(arg->type()->kind() == TypeKind::Literal) {
                check_literal_types(arg, param, param->type);
            } else {
                print_type_mismatch(arg, param, param->type, "Expected function parameter",
                                    "Actual function argument");
            }
        }
    }
}

void IndexOp::check_types()
{
    if(base_expr->kind() != ExpressionKind::LValue) {
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
    const Type* index_expr_type = index_expr->type();
    if(index_expr_type != arr_type->index_type) {
        if(index_expr_type->kind() == TypeKind::Literal) {
            check_literal_types(index_expr.get(), base_expr.get(), arr_type->index_type);
        } else {
            print_type_mismatch(index_expr.get(), base_expr.get(), arr_type->index_type,
                                "Expected array index",
                                "Actual");
        }
    }
}

void InitList::check_types()
{
    if(lvalue == nullptr) {
        Error(line_num())
            .raise("Initializer list used in incorrect context."
                   " Initializer lists can only be used to initialize/assign"
                   " a variable or constant of array/record types, not as part"
                   " of a larger expression");
    }

    if(lvalue->type->kind() == TypeKind::Array) {
        auto* arr_type = static_cast<ArrayType*>(lvalue->type);
        // TODO: zero-initialize all other indices
        if(values.size() > arr_type->index_type->range.size()) {
            Error(line_num()).put(" Array ").put(lvalue->type)
                .put(" expects at most ").put(arr_type->index_type->range.size())
                .put(" values, bit this initializer list provides ")
                .put(values.size()).raise(" value(s)");
        }

        for(auto& value : values) {
            value->check_types();
            auto* val_type = value->type();
            if(val_type != arr_type->element_type) {
                if(val_type->kind() == TypeKind::Literal) {
                    check_literal_types(value.get(), lvalue, arr_type->element_type);
                } else {
                    print_type_mismatch(value.get(), lvalue, arr_type->element_type,
                                        "Expected initializer list item",
                                        "Actual item");
                }
            }
        }
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
    return false;
}

bool Initialization::check_types(Checker&)
{
    if(expression == nullptr)
        return false;
    expression->check_types();
    if(expression->type() != lvalue->type) {
        if(expression->type()->kind() == TypeKind::Literal) {
            check_literal_types(expression.get(), lvalue.get(), lvalue->type);
        } else {
            print_type_mismatch(expression.get(), lvalue.get(), lvalue->type);
        }
    }
    return false;
}

bool Assignment::check_types(Checker&)
{
    expression->check_types();
    if(expression->type() != lvalue->type) {
        if(expression->type()->kind() == TypeKind::Literal) {
            check_literal_types(expression.get(), lvalue, lvalue->type);
        } else {
            print_type_mismatch(expression.get(), lvalue, lvalue->type);
        }
    }
    return false;
}

static bool is_bool_condition(const Expression* condition)
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
    expression->check_types();

    const Type* return_type = expression->type();
    const BBFunction* curr_funct = checker.m_curr_funct;
    if(curr_funct->return_type != return_type) {
        if(return_type->kind() == TypeKind::Literal) {
            check_literal_types(expression.get(), curr_funct, curr_funct->return_type);
        } else {
            if(curr_funct->return_type == &Type::Void) {
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
    }
    return true;
}

void Checker::run()
{
    for(auto& var : m_global_vars) {
        var->check_types(*this);
        if(var->expression != nullptr) {
            switch(var->expression->kind()) {
            case ExpressionKind::CharLiteral:
            case ExpressionKind::IntLiteral:
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
            if(!always_returns && m_curr_funct->return_type != &Type::Void) {
                Error().put("Function").quote(m_curr_funct->name)
                    .raise("does not return in all cases");
            }
        }
    }
}
