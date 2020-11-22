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
        actual->expression->check_types(statement);
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
       && (right_cat != TypeCategory::Literal)) {
        return left == &LiteralType::Int && right_cat == TypeCategory::Range;
    } else if((left_cat != TypeCategory::Literal)
              && (right_cat == TypeCategory::Literal)) {
        return right == &LiteralType::Int && left_cat == TypeCategory::Range;
    } else {
        return false;
    }
}

// No need to typecheck for literal/lvalue expressions since they aren't composite,
// so each of their check_types() member functions are empty, defined in header

// TODO: check that unary operators are valid for their values
void UnaryExpression::check_types(const Statement*) const
{}

// TODO: check if this binary op is legal for this type
void BinaryExpression::check_types(const Statement* statement) const
{
    // Ensure the sub-expressions are correct first
    left->check_types(statement);
    right->check_types(statement);
    // Check that the types of both sides of the operator match
    if(left->type() != right->type() && !type_matches_literal(left->type(), right->type())) {
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
        if(arg->type() != param->type
           && !type_matches_literal(arg->type(), param->type)) {
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
        }
    }
}

void Checker::run() const
{
    for(const auto *function : m_functions) {
        for(const auto *statement : function->statements) {
            check_types(statement);
        }
    }
}
