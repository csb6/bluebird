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
#include <iostream>

Checker::Checker(const std::vector<Function*>& functions,
                 const std::vector<RangeType*>& types,
                 const std::vector<Initialization*>& global_vars)
    : m_functions(functions), m_types(types), m_global_vars(global_vars)
{}

static void print_error(const Expression* expr, const char* message)
{
    std::cerr << "ERROR: In expression starting at line " << expr->line_num()
              << ":\n" << message;
}

static void print_error(unsigned int line_num, const char* message)
{
    std::cerr << "ERROR: line " << line_num << ":\n " << message << "\n";
}

template<typename Other>
static void print_type_mismatch(const Expression* expr,
                                const Other* other, const Type* other_type,
                                const char* other_label = "",
                                const char* expr_label = "Expression",
                                const TokenType* op = nullptr)
{
    print_error(expr, " Types don't match:\n  ");

    // Left
    std::cerr << expr_label << ": ";
    expr->print(std::cerr);
    std::cerr << '\t';
    expr->type()->print(std::cerr);
    // Op
    if(op != nullptr) {
        std::cerr << "  Operator: " << *op << "\n";
    }
    // Right
    std::cerr << "  ";
    if(other_label[0] != '\0') {
        std::cerr << other_label << ": ";
    }
    other->print(std::cerr);
    std::cerr << '\t';
    other_type->print(std::cerr);
}

static void nonbool_condition_error(const Expression* condition,
                                    const char* statement_name)
{
    print_error(condition, " Exprected boolean condition for this ");
    std::cerr << statement_name << ", but instead found:\n  Expression: ";
    condition->print(std::cerr);
    std::cerr << '\t';
    condition->type()->print(std::cerr);
}

// No need to typecheck for literal/lvalue expressions since they aren't composite,
// so each of their check_types() member functions are empty, defined in header

template<typename Other>
static void check_literal_types(Expression* literal, const Other* other,
                                const Type* other_type)
{
    if(other_type->category() == TypeCategory::Range) {
        auto* range_type = static_cast<const RangeType*>(other_type);
        const Range& range = range_type->range;
        switch(literal->kind()) {
        case ExpressionKind::IntLiteral: {
            auto* int_literal = static_cast<IntLiteral*>(literal);
            if(!range.contains(int_literal->value)) {
                print_error(int_literal, " Integer Literal `");
                int_literal->print(std::cerr);
                std::cerr << "` is not in the range of:\n  ";
                range_type->print(std::cerr);
                std::cerr << " so it cannot be used with:\n  ";
                other->print(std::cerr);
                std::cerr << "\n";
                exit(1);
            }
            // Literals (if used with a compatible range type) take on the type of what
            // they are used with
            int_literal->actual_type = other_type;
            break;
        }
        case ExpressionKind::CharLiteral: {
            auto* char_literal = static_cast<CharLiteral*>(literal);
            if(other_type != &RangeType::Character) {
                print_error(char_literal, " Character Literal ");
                char_literal->print(std::cerr);
                std::cerr << " cannot be used with the non-character type:\n  ";
                other_type->print(std::cerr);
                std::cerr << "\n";
                exit(1);
            }
            char_literal->actual_type = other_type;
            break;
        }
        default:
            print_error(literal, " Expected a literal type, but instead found expression: ");
            literal->print(std::cerr);
            std::cerr << "\n Which has type:\n  ";
            range_type->print(std::cerr);
            std::cerr << "\n";
            exit(1);
        }
    } else {
        print_type_mismatch(literal, other, other_type, "Used with", "Literal");
        exit(1);
    }
}

// TODO: check that unary operators are valid for their values
void UnaryExpression::check_types()
{}

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
        if(right_type->category() == TypeCategory::Literal) {
            check_literal_types(right.get(), left.get(), left_type);
        } else if(left_type->category() == TypeCategory::Literal) {
            check_literal_types(left.get(), right.get(), right_type);
        } else {
            print_type_mismatch(left.get(), right.get(), right_type, "Right", "Left", &op);
            exit(1);
        }
    }
}

void FunctionCall::check_types()
{
    if(arguments.size() != function->parameters.size()) {
        print_error(this, " Function `");
        std::cerr << name << "` expects " << function->parameters.size()
                  << " arguments, but " << arguments.size() << " were provided\n";
        exit(1);
    }
    const size_t arg_count = arguments.size();
    for(size_t i = 0; i < arg_count; ++i) {
        Expression* arg = arguments[i].get();
        // Ensure each argument expression is internally typed correctly
        arg->check_types();
        // Make sure each arg type matches corresponding parameter type
        const LValue* param = function->parameters[i];
        if(arg->type() != param->type) {
            if(arg->type()->category() == TypeCategory::Literal) {
                check_literal_types(arg, param, param->type);
            } else {
                print_type_mismatch(arg, param, param->type, "Expected function parameter",
                                    "Actual function argument");
                exit(1);
            }
        }
    }
}

void BasicStatement::check_types(Checker&)
{
    expression->check_types();
}

void Initialization::check_types(Checker&)
{
    if(expression == nullptr)
        return;
    expression->check_types();
    if(expression->type() != lvalue->type) {
        if(expression->type()->category() == TypeCategory::Literal) {
            check_literal_types(expression.get(), lvalue, lvalue->type);
        } else {
            print_type_mismatch(expression.get(), lvalue, lvalue->type);
            exit(1);
        }
    }
}

void Assignment::check_types(Checker&)
{
    expression->check_types();
    if(expression->type() != lvalue->type) {
        if(expression->type()->category() == TypeCategory::Literal) {
            check_literal_types(expression.get(), lvalue, lvalue->type);
        } else {
            print_type_mismatch(expression.get(), lvalue, lvalue->type);
            exit(1);
        }
    }
}

void IfBlock::check_types(Checker& checker)
{
    condition->check_types();
    if(condition->type() != &Type::Bool) {
        nonbool_condition_error(condition.get(), "if-statement");
        exit(1);
    }
    Block::check_types(checker);
    if(else_or_else_if != nullptr) {
        else_or_else_if->check_types(checker);
    }
}

void Block::check_types(Checker& checker)
{
    for(auto* stmt : statements) {
        stmt->check_types(checker);

        if(stmt->kind() == StatementKind::Return && stmt != statements.back()) {
            auto* ret_stmt = static_cast<ReturnStatement*>(stmt);
            print_error(ret_stmt->expression->line_num(),
                "Statements after return statement:");
            std::cerr << "  ";
            ret_stmt->print(std::cerr);
            std::cerr << " are unreachable\n";
            exit(1);
        }
    }
}

void WhileLoop::check_types(Checker& checker)
{
    condition->check_types();
    if(condition->type() != &Type::Bool) {
        nonbool_condition_error(condition.get(), "while-loop");
        exit(1);
    }
    Block::check_types(checker);
}

void ReturnStatement::check_types(Checker& checker)
{
    expression->check_types();

    const Type* return_type = expression->type();
    const BBFunction* curr_funct = checker.m_curr_funct;
    if(curr_funct->return_type != return_type) {
        if(return_type->category() == TypeCategory::Literal) {
            check_literal_types(expression.get(), curr_funct, curr_funct->return_type);
        } else {
            print_error(expression.get(), " Wrong return type: ");
            return_type->print(std::cerr);
            if(curr_funct->return_type == &Type::Void) {
                std::cerr << " Did not expect void function `" << curr_funct->name
                          << "` to return something\n";
            } else {
                std::cerr << " Expected return type for this function: ";
                curr_funct->return_type->print(std::cerr);
                std::cerr << "\n";
            }
            exit(1);
        }
    }
}

void Checker::run()
{
    for(Initialization* var : m_global_vars) {
        var->check_types(*this);
        if(var->expression != nullptr) {
            switch(var->expression->kind()) {
            case ExpressionKind::CharLiteral:
            case ExpressionKind::IntLiteral:
                break;
            default:
                print_error(var->expression.get(),
                            " Global variables/constants can only be initialized"
                            " to integer or character literals\n");
                exit(1);
            }
        }
    }

    for(Function* function : m_functions) {
        if(function->kind() == FunctionKind::Normal) {
            m_curr_funct = static_cast<BBFunction*>(function);
            m_curr_funct->body.check_types(*this);
        }
    }
}
