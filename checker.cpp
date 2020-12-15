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

// No need to typecheck for literal/lvalue expressions since they aren't composite,
// so each of their check_types() member functions are empty, defined in header

// TODO: check that unary operators are valid for their values
void UnaryExpression::check_types(const Statement*) const
{}

template<typename Other>
static void check_literal_types(const IntLiteral* literal, const Other* other,
                                const Type* other_type, const Statement* stmt)
{
    if(other_type->category() == TypeCategory::Range) {
        auto* range_type = static_cast<const RangeType*>(other_type);
        const Range& range = range_type->range;
        if(!range.contains(literal->value)) {
            std::cerr << "In statement starting at line " << stmt->line_num
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
        std::cerr << "In statement starting at line " << stmt->line_num
                  << ":\n Types don't match:\n  IntLiteral: ";
        literal->print(std::cerr);
        std::cerr << "\n  Used with: ";
        other->print(std::cerr);
        std::cerr << '\t';
        other_type->print(std::cerr);
        exit(1);
    }
}

// TODO: check if this binary op is legal for this type
void BinaryExpression::check_types(const Statement* statement) const
{
    // Ensure the sub-expressions are correct first
    left->check_types(statement);
    right->check_types(statement);
    // Check that the types of both sides of the operator match
    if(left->type() != right->type()) {
        std::cerr << "In statement starting at line " << statement->line_num
                  << ":\n Types don't match:\n  Left: ";
        left->print(std::cerr);
        std::cerr << '\t';
        left->type()->print(std::cerr);
        std::cerr << "  Operator: " << op << "\n  Right: ";
        right->print(std::cerr);
        std::cerr << '\t';
        right->type()->print(std::cerr);
        exit(1);
    } else if(right->kind() == ExpressionKind::IntLiteral) {
        check_literal_types(static_cast<const IntLiteral*>(right.get()), left.get(),
                            left->type(), statement);
    } else if(left->kind() == ExpressionKind::IntLiteral) {
        check_literal_types(static_cast<const IntLiteral*>(left.get()), right.get(),
                            right->type(), statement);
    }
}

void FunctionCall::check_types(const Statement* statement) const
{
    if(arguments.size() != function->parameters.size()) {
        std::cerr << "In statement starting at line " << statement->line_num
                  << ":\n Function `" << name << "` expects "
                  << function->parameters.size() << " arguments, but "
                  << arguments.size() << " were provided\n";
        exit(1);
    }
    const size_t arg_count = arguments.size();
    for(size_t i = 0; i < arg_count; ++i) {
        const Expression* arg = arguments[i].get();
        // Ensure each argument expression is internally typed correctly
        arg->check_types(statement);
        // Make sure each arg type matches corresponding parameter type
        const LValue* param = function->parameters[i];
        if(arg->type() != param->type) {
            std::cerr << "In statement starting at line " << statement->line_num
                      << ":\n Argument type doesn't match expected type:\n"
                         "  Argument: ";
            arg->print(std::cerr);
            std::cerr << '\t';
            arg->type()->print(std::cerr);
            std::cerr << "  Expected: ";
            param->print(std::cerr);
            std::cerr << '\t';
            param->type->print(std::cerr);
            exit(1);
        } else if(arg->kind() == ExpressionKind::IntLiteral) {
            check_literal_types(static_cast<const IntLiteral*>(arg), param,
                                param->type, statement);
        }
    }
}

void BasicStatement::check_types() const
{
    expression->check_types(this);
}

void Initialization::check_types() const
{
    if(expression == nullptr)
        return;
    expression->check_types(this);
}

void Assignment::check_types() const
{
    expression->check_types(this);
}

void IfBlock::check_types() const
{
    condition->check_types(this);
    for(auto* stmt : statements) {
        stmt->check_types();
    }
}

void Checker::run() const
{
    for(const auto *function : m_functions) {
        for(const auto *statement : function->statements) {
            statement->check_types();
        }
    }
}
