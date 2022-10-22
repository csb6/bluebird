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
#include "ast.h"
#include <limits>

size_t ArrayType::bit_size() const
{
    // TODO: account for padding of elements
    return index_type->range.size() * element_type->bit_size();
}


IntRange::IntRange(multi_int lower, multi_int upper, bool inclusive)
    : lower_bound(std::move(lower)), upper_bound(std::move(upper)), is_signed(lower_bound.is_negative())
{
    if(!inclusive) {
        upper_bound -= 1;
    }

    if(is_signed) {
        if(lower_bound.bits_needed() > upper_bound.bits_needed()) {
            multi_int lower_double{lower_bound};
            lower_double *= 2;
            lower_double += 1;
            bit_size = lower_double.bits_needed();
        } else {
            multi_int upper_double{upper_bound};
            upper_double *= 2;
            upper_double += 1;
            bit_size = upper_double.bits_needed();
        }
    } else {
        bit_size = upper_bound.bits_needed();
    }
}

bool IntRange::contains(const multi_int& value) const
{
    return value >= lower_bound && value <= upper_bound;
}

unsigned long int IntRange::size() const
{
    multi_int low_copy{lower_bound};
    low_copy.negate();
    low_copy += upper_bound;
    low_copy += 1;
    return to_int(std::move(low_copy));
}

bool FloatRange::contains(double value) const
{
    if(is_inclusive) {
        return value >= lower_bound && value <= upper_bound;
    }

    return value >= lower_bound && value < upper_bound;
}

const Type* VariableExpression::type() const
{
    return variable->type;
}

const Type* UnaryExpression::type() const
{
    if(actual_type == nullptr) {
        return right->type();
    }
    return actual_type;
}

const Type* BinaryExpression::type() const
{
    if(is_comparison_op(op) || is_logical_op(op)) {
        return &EnumType::Boolean;
    }
    // If a literal and a typed expression of some sort
    // are in this expression, want to return the type of
    // the typed part (literals implicitly convert to that type)
    switch(left->kind()) {
    case ExprKind::StringLiteral:
    case ExprKind::CharLiteral:
    case ExprKind::IntLiteral:
    case ExprKind::FloatLiteral:
        return right->type();
    default:
        return left->type();
    }
}

const std::string& FunctionCall::name() const { return definition->name; }

const Type* FunctionCall::type() const { return definition->return_type; }

const Type* IndexedExpr::type() const
{
    const Type* base_type = base_expr->type();
    if(base_type->kind() == TypeKind::Array) {
        return as<ArrayType>(base_type)->element_type;
    }
    return &Type::Void;
}

unsigned int ReturnStatement::line_num() const
{
    if(expression.get() != nullptr) {
        return expression->line_num();
    }
    return line;
}

const Type Type::Void{"VoidType"};
const Type Type::String{"StringLiteral"};

const EnumType EnumType::Boolean{"Boolean"};

const LiteralType LiteralType::Char{"CharLiteral"};
const LiteralType LiteralType::Int{"IntLiteral"};
const LiteralType LiteralType::Float{"FloatLiteral"};
const LiteralType LiteralType::Bool{"BoolLiteral"};
const LiteralType LiteralType::InitList{"InitList"};

// -2^31 thru 2^31-1 (same as the GNAT Ada compiler defines it)
const IntRangeType IntRangeType::Integer{"Integer", IntRange{multi_int{"-2147483648"}, multi_int{"2147483647"}, true}};
const IntRangeType IntRangeType::Character{"Character", IntRange{multi_int{"0"}, multi_int{"255"}, false}};
const FloatRangeType FloatRangeType::Float{"Float", FloatRange{std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), true}};
