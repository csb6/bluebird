/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020  Cole Blakley

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
                 const std::vector<RangeType*>& types)
    : m_functions(functions), m_types(types)
{}

template<typename Other>
static void print_type_mismatch(const Expression* expr,
                                const Other* other, const Type* other_type,
                                const char* other_label = "",
                                const char* expr_label = "Expression",
                                const TokenType* op = nullptr)
{
    std::cerr << "ERROR: In expression starting at line " << expr->line_num()
              << ":\n Types don't match:\n  ";

    // Left
    std::cerr << expr_label << ": ";
    expr->print(std::cerr);
    std::cerr << '\t';
    expr->type()->print(std::cerr);
    // Op
    if(op != nullptr) {
        std::cerr << "  Operator: " << *op << '\n';
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

// No need to typecheck for literal/lvalue expressions since they aren't composite,
// so each of their check_types() member functions are empty, defined in header

template<typename Other>
static void check_literal_types(const IntLiteral* literal, const Other* other,
                                const Type* other_type)
{
    if(other_type->category() == TypeCategory::Range) {
        auto* range_type = static_cast<const RangeType*>(other_type);
        const Range& range = range_type->range;
        if(!range.contains(literal->value)) {
            std::cerr << "ERROR: In expression starting at line " << literal->line
                      << ":\n Integer literal `";
            literal->print(std::cerr);
            std::cerr << "` is not in the range of:\n  ";
            range_type->print(std::cerr);
            std::cerr << " so it cannot be used with:\n  ";
            other->print(std::cerr);
            std::cerr << "\n Which has type:\n  ";
            range_type->print(std::cerr);
            std::cerr << "\n";
            exit(1);
        }
    } else {
        print_type_mismatch(literal, other, other_type, "Used with", "IntLiteral");
        exit(1);
    }
}

// TODO: check that unary operators are valid for their values
void UnaryExpression::check_types() const
{}

// TODO: check if this binary op is legal for this type
void BinaryExpression::check_types() const
{
    // Ensure the sub-expressions are correct first
    left->check_types();
    right->check_types();
    const Type* left_type = left->type();
    const Type* right_type = right->type();
    // Check that the types of both sides of the operator match
    if(left_type != right_type) {
        print_type_mismatch(left.get(), right.get(), right_type, "Right", "Left", &op);
        exit(1);
    } else if(right->kind() == ExpressionKind::IntLiteral) {
        check_literal_types(static_cast<const IntLiteral*>(right.get()), left.get(),
                            left_type);
    } else if(left->kind() == ExpressionKind::IntLiteral) {
        check_literal_types(static_cast<const IntLiteral*>(left.get()), right.get(),
                            right_type);
    }
}

void FunctionCall::check_types() const
{
    if(arguments.size() != function->parameters.size()) {
        std::cerr << "ERROR: In expression starting at line " << line
                  << ":\n Function `" << name << "` expects "
                  << function->parameters.size() << " arguments, but "
                  << arguments.size() << " were provided\n";
        exit(1);
    }
    const size_t arg_count = arguments.size();
    for(size_t i = 0; i < arg_count; ++i) {
        const Expression* arg = arguments[i].get();
        // Ensure each argument expression is internally typed correctly
        arg->check_types();
        // Make sure each arg type matches corresponding parameter type
        const LValue* param = function->parameters[i];
        if(arg->type() != param->type) {
            print_type_mismatch(arg, param, param->type,
                                "Expected function parameter", "Actual function argument");
            exit(1);
        } else if(arg->kind() == ExpressionKind::IntLiteral) {
            check_literal_types(static_cast<const IntLiteral*>(arg), param,
                                param->type);
        }
    }
}

void BasicStatement::check_types() const
{
    expression->check_types();
}

void Initialization::check_types() const
{
    if(expression == nullptr)
        return;
    expression->check_types();
    if(expression->type() != lvalue->type) {
        print_type_mismatch(expression.get(), lvalue, lvalue->type);
        exit(1);
    }
}

void Assignment::check_types() const
{
    expression->check_types();
    if(expression->type() != lvalue->type) {
        print_type_mismatch(expression.get(), lvalue, lvalue->type);
        exit(1);
    }
}

void IfBlock::check_types() const
{
    condition->check_types();
    if(condition->type() != &Type::Bool) {
        std::cerr << "ERROR: In statement starting at line " << condition->line_num()
                  << ":\n Expected boolean condition for this if-statement,"
                     " but instead found:\n  Expression: ";
        condition->print(std::cerr);
        std::cerr << '\t';
        condition->type()->print(std::cerr);
        exit(1);
    }
    Block::check_types();
    if(else_or_else_if != nullptr) {
        else_or_else_if->check_types();
    }
}

void Block::check_types() const
{
    for(auto* stmt : statements) {
        stmt->check_types();
    }
}

void Checker::run() const
{
    for(const auto *function : m_functions) {
        if(function->kind() == FunctionKind::Normal) {
            for(const auto *statement
                    : static_cast<const BBFunction*>(function)->statements) {
                statement->check_types();
            }
        }
    }
}
