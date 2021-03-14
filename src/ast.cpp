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
#include "ast.h"
#include <iomanip> // for setprecision()

void Type::print(std::ostream& output) const
{
    output << "Type: " << name;
}

void RangeType::print(std::ostream& output) const
{
    output << "Type: " << name << " Range: " << range;
}

void ArrayType::print(std::ostream& output) const
{
    output << "Type: " << name << " Array(";
    index_type->print(output);
    output << ") of ";
    element_type->print(output);
}

void RefType::print(std::ostream& output) const
{
    output << "Type: " << name << " Ref ";
    inner_type->print(output);
}

Range::Range(const multi_int& lower, const multi_int& upper)
    : lower_bound(lower), upper_bound(upper),
      is_signed(lower.is_negative())
{
    if(is_signed) {
        if(lower.bits_needed() > upper.bits_needed()) {
            multi_int lower_double{lower};
            lower_double *= 2;
            lower_double += 1;
            bit_size = lower_double.bits_needed();
        } else {
            multi_int upper_double{upper};
            upper_double *= 2;
            upper_double += 1;
            bit_size = upper_double.bits_needed();
        }
    } else {
        bit_size = upper.bits_needed();
    }
}

bool Range::contains(const multi_int& value) const
{
    return value >= lower_bound && value <= upper_bound;
}

unsigned long int Range::size() const
{
    multi_int low_copy{lower_bound};
    low_copy.negate();
    low_copy += upper_bound;
    low_copy += 1;
    return to_int(std::move(low_copy));
}

std::ostream& operator<<(std::ostream& output, const Range& range)
{
    output << "(" << range.lower_bound << ", " << range.upper_bound << ")";
    return output;
}

const Type* LValueExpression::type() const
{
    return lvalue->type;
}

void LValueExpression::print(std::ostream& output) const
{
    output << lvalue->name;
}

void RefExpression::print(std::ostream& output) const
{
    output << "Ref to: ";
    lvalue->print(output);
}

void StringLiteral::print(std::ostream& output) const
{
    output << "\"";
    print_unescape(value, output);
    output << "\"";
}

void CharLiteral::print(std::ostream& output) const
{
    output << "'";
    print_unescape(value, output);
    output << "'";
}

void IntLiteral::print(std::ostream& output) const
{
    output << value;
}

void BoolLiteral::print(std::ostream& output) const
{
    output << value;
}

void FloatLiteral::print(std::ostream& output) const
{
    output << std::setprecision(10);
    output << value;
}

void UnaryExpression::print(std::ostream& output) const
{
    output << "(" << op << " ";
    right->print(output);
    output << ")";
}

const Type* BinaryExpression::type() const
{
    if(is_bool_op(op)) {
        return &EnumType::Boolean;
    } else {
        // If a literal and a typed expression of some sort
        // are in this expression, want to return the type of
        // the typed part (literals implicitly convert to that type)
        switch(left->kind()) {
        case ExpressionKind::StringLiteral: case ExpressionKind::CharLiteral:
        case ExpressionKind::IntLiteral:    case ExpressionKind::FloatLiteral:
            return right->type();
        default:
            return left->type();
        }
    }
}

void BinaryExpression::print(std::ostream& output) const
{
    output << "(";
    left->print(output);
    output << " " << op << " ";
    right->print(output);
    output << ")";
}

const std::string& FunctionCall::name() const { return definition->name; }

const Type* FunctionCall::type() const { return definition->return_type; }

void FunctionCall::print(std::ostream& output) const
{
    output << name() << "(";
    for(const auto& argument : arguments) {
        argument->print(output);
        output << ", ";
    }
    output << ")";
}

const Type* IndexOp::type() const
{
    const Type* base_type = base_expr->type();
    if(base_type->kind() == TypeKind::Array) {
        auto* arr_type = static_cast<const ArrayType*>(base_type);
        return arr_type->element_type;
    } else {
        return &Type::Void;
    }
}

void IndexOp::print(std::ostream& output) const
{
    output << "(";
    base_expr->print(output);
    output << ")[";
    index_expr->print(output);
    output << "]";
}

const Type* InitList::type() const
{
    if(lvalue == nullptr) {
        return &Type::Void;
    } else {
        return lvalue->type;
    }
}

void InitList::print(std::ostream& output) const
{
    output << "{ ";
    for(const auto& each : values) {
        each->print(output);
        output << ", ";
    }
    output << "}";
}

void BasicStatement::print(std::ostream& output) const
{
    if(expression) {
        output << "Statement: ";
        expression->print(output);
    } else {
        output << "Empty Statement";
    }
}

void Initialization::print(std::ostream& output) const
{
    output << "Initialize ";
    lvalue->print(output);
    output << " = ";
    if(expression) {
        expression->print(output);
    } else {
        output << "Empty Statement";
    }
}

void Assignment::print(std::ostream& output) const
{
    output << "Assign ";
    lvalue->print(output);
    output << " = ";
    expression->print(output);
}

void Block::print(std::ostream& output) const
{
    output << "Block:";
    for(const auto& each : statements) {
        output << "\n";
        each->print(output);
    }
}

void IfBlock::print(std::ostream& output) const
{
    output << "If Block:\n";
    output << "Condition: ";
    condition->print(output);
    output << "\n";
    Block::print(output);
    if(else_or_else_if != nullptr) {
        output << "\nElse ";
        else_or_else_if->print(output);
    }
}

void WhileLoop::print(std::ostream& output) const
{
    output << "While Loop:\n";
    output << "Condition: ";
    condition->print(output);
    output << "\n";
    Block::print(output);
}

unsigned int ReturnStatement::line_num() const
{
    if(expression.get() != nullptr) {
        return expression->line_num();
    } else {
        return line;
    }
}

void ReturnStatement::print(std::ostream& output) const
{
    output << "Return: ";
    if(expression.get() != nullptr) {
        expression->print(output);
    } else {
        output << "Empty statement";
    }
}

void NamedLValue::print(std::ostream& output) const
{
    if(is_mutable) {
        output << "Variable: ";
    } else {
        output << "Constant: ";
    }
    output << name;
}

void IndexLValue::print(std::ostream& output) const
{
    if(is_mutable) {
        output << "Variable: ";
    } else {
        output << "Constant: ";
    }
    array_access->print(output);
}

void BBFunction::print(std::ostream& output) const
{
    output << "Function: " << name;
    if(return_type != nullptr) {
        output << "\tReturn ";
        return_type->print(output);
    }
    output << "\nParameters:\n";
    for(const auto& param : parameters) {
        param->print(output);
        output << "\n";
    }
    output << "Body:\n";
    body.print(output);
}

void BuiltinFunction::print(std::ostream& output) const
{
    output << "Built-In Function: " << name;
}

Type Type::Void{"VoidType"};
Type Type::String{"StringLiteral"};
Type Type::Float{"FloatLiteral"};

EnumType EnumType::Boolean{"Boolean"};

const LiteralType LiteralType::Char{"CharLiteral"};
const LiteralType LiteralType::Int{"IntLiteral"};
const LiteralType LiteralType::Bool{"BoolLiteral"};

// -2^31 thru 2^31-1 (same as the GNAT Ada compiler defines it)
RangeType RangeType::Integer{"Integer", multi_int{"-2147483648"}, multi_int{"2147483647"}};
RangeType RangeType::Character{"Character", multi_int{"0"}, multi_int{"255"}};
