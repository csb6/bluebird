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
    output << "Type: " << name << "\n";
}

void RangeType::print(std::ostream& output) const
{
    output << "Type: " << name << " Range: " << range << "\n";
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

std::ostream& operator<<(std::ostream& output, const Range& range)
{
    output << '(' << range.lower_bound << ", " << range.upper_bound << ')';
    return output;
}

const Type* LValueExpression::type() const
{
    return lvalue->type;
}

void LValueExpression::print(std::ostream& output) const
{
    output << name;
}

UnaryExpression::UnaryExpression(TokenType oper, Expression* r)
    : op(oper), right(r)
{}

void StringLiteral::print(std::ostream& output) const
{
    output << '"';
    print_unescape(value, output);
    output << '"';
}

void CharLiteral::print(std::ostream& output) const
{
    output << '\'';
    print_unescape(value, output);
    output << '\'';
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
    output << '(' << op << ' ';
    right->print(output);
    output << ')';
}

BinaryExpression::BinaryExpression(Expression* l, TokenType oper,
                                   Expression* r)
    : left(l), op(oper), right(r)
{}

const Type* BinaryExpression::type() const
{
    if(is_bool_op(op)) {
        return &Type::Boolean;
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
    output << '(';
    left->print(output);
    output << ' ' << op << ' ';
    right->print(output);
    output << ')';
}

const Type* FunctionCall::type() const { return function->return_type; }

void FunctionCall::print(std::ostream& output) const
{
    output << name << '(';
    for(const auto& argument : arguments) {
        argument->print(output);
        output << ", ";
    }
    output << ')';
}

void BasicStatement::print(std::ostream& output) const
{
    if(expression) {
        output << "Statement:\n";
        expression->print(output);
        output << "\n";
    } else {
        output << "Empty Statement\n";
    }
}

void Initialization::print(std::ostream& output) const
{
    output << "Initialize ";
    lvalue->print(output);
    output << " = ";
    if(expression) {
        output << "Expression: ";
        expression->print(output);
        output << "\n";
    } else {
        output << "Empty Statement\n";
    }
}

void Assignment::print(std::ostream& output) const
{
    output << "Assign ";
    lvalue->print(output);
    output << " = ";
    expression->print(output);
    output << "\n";
}

void Block::print(std::ostream& output) const
{
    output << "Block:\n";
    output << "Block Statements:\n";
    for(const auto& each : statements) {
        each->print(output);
    }
    output << "\n";
}

void IfBlock::print(std::ostream& output) const
{
    output << "If Block:\n";
    output << "Condition: ";
    condition->print(output);
    output << "\n";
    Block::print(output);
    if(else_or_else_if != nullptr) {
        output << "Else ";
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

void ReturnStatement::print(std::ostream& output) const
{
    output << "Return: ";
    expression->print(output);
    output << "\n";
}

void LValue::print(std::ostream& output) const
{
    if(is_mutable) {
        output << "Variable: ";
    } else {
        output << "Constant: ";
    }

    output << name;
}

void BBFunction::print(std::ostream& output) const
{
    output << "Function: " << name;
    if(return_type != nullptr) {
        output << "\tReturn ";
        return_type->print(output);
    } else {
        output << "\n";
    }
    output << "Parameters:\n";
    for(const auto& param : parameters) {
        param->print(output);
        output << "\n";
    }
    output << "Body:\n";
    body.print(output);
}

void BuiltinFunction::print(std::ostream& output) const
{
    output << "Built-In Function: " << name << "\n";
}

Type Type::Void{"VoidType"};
Type Type::String{"StringLiteral"};
Type Type::Float{"FloatLiteral"};
Type Type::Boolean{"Boolean"};

const LiteralType LiteralType::Char{"CharLiteral"};
const LiteralType LiteralType::Int{"IntLiteral"};
const LiteralType LiteralType::Bool{"BoolLiteral"};

// -2^31 thru 2^31-1 (same as the GNAT Ada compiler defines it)
RangeType RangeType::Integer{"Integer", multi_int{"-2147483648"}, multi_int{"2147483647"}};
RangeType RangeType::Character{"Character", multi_int{"0"}, multi_int{"255"}};
