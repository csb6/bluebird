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

void Checker::check_types(const Statement* statement) const
{
    switch(statement->kind()) {
    case StatementKind::Basic: {
        auto* actual = static_cast<const BasicStatement*>(statement);
        check_types(statement, actual->expression.get());
        break;
    }
    case StatementKind::Initialization:
        break;
    case StatementKind::IfBlock:
        break;
    }
}

bool type_matches_literal(const Type *left, const Type *right)
{
    const TypeCategory left_cat = left->category();
    const TypeCategory right_cat = right->category();
    // TODO: check that the literal is in the range of the non-literal type
    if((left_cat == TypeCategory::Literal)
       & (right_cat != TypeCategory::Literal)) {
        return left == &LiteralType::Int && right_cat == TypeCategory::Range;
    } else if((left_cat != TypeCategory::Literal)
              & (right_cat == TypeCategory::Literal)) {
        return right == &LiteralType::Int && left_cat == TypeCategory::Range;
    } else {
        return false;
    }
}

void Checker::check_types(const Statement* statement, const Expression* expression) const
{
    switch(expression->kind()) {
    case ExpressionKind::Binary: {
        // TODO: check if this binary op is legal for this type
        auto* actual = static_cast<const BinaryExpression*>(expression);
        // Ensure the sub-expressions are correct first
        check_types(statement, actual->left.get());
        check_types(statement, actual->right.get());
        // Check that the types of both sides of the operator match
        if(actual->left->type() != actual->right->type()
           && !type_matches_literal(actual->left->type(), actual->right->type())) {
            std::cerr << "In statement starting at line " << statement->line_num
                      << ": \nTypes don't match:\n  Left: ";
            actual->left->print(std::cerr);
            std::cerr << '\t';
            actual->left->type()->print(std::cerr);
            std::cerr << "  Operator: " << actual->op << "\n  Right: ";
            actual->right->print(std::cerr);
            std::cerr << '\t';
            actual->right->type()->print(std::cerr);
            exit(1);
        }
        break;
    }
    case ExpressionKind::FunctionCall:
        break;
    case ExpressionKind::Unary:
        // Check that the unary op is legal for this type
        break;
    default:
        // For terminal nodes, no need to typecheck, since not composite
        break;
    }
}

void Checker::run()
{
    for(const auto *function : m_functions) {
        for(const auto &statement : function->statements) {
            check_types(statement);
            // TODO: add more checks for this statement here
        }
    }
}
