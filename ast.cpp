#include "ast.h"
#include <iomanip> // for setprecision()

Range::Range(const multi_int& lower, const multi_int& upper)
    : lower_bound(lower), upper_bound(upper),
      bit_size(std::max(lower.bits_needed(), upper.bits_needed()))
{}

IntLiteral::IntLiteral(const std::string& v)
    : value(v), bit_size(value.bits_needed())
{}

void LValueExpression::print(std::ostream& output) const
{
    output << name;
}

UnaryExpression::UnaryExpression(TokenType oper, Expression* r)
    : op(oper), right(r)
{}

void StringLiteral::print(std::ostream& output) const
{
    output << value;
}

void CharLiteral::print(std::ostream& output) const
{
    output << value;
}

void IntLiteral::print(std::ostream& output) const
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

void BinaryExpression::print(std::ostream& output) const
{
    output << '(';
    left->print(output);
    output << ' ' << op << ' ';
    right->print(output);
    output << ')';
}

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
        output << '\n';
    } else {
        output << "Empty Statement\n";
    }
}

void Initialization::print(std::ostream& output) const
{
    output << "Initialization ";
    BasicStatement::print(output);
}

void IfBlock::print(std::ostream& output) const
{
    output << "If Block:\n";
    output << "Condition: ";
    condition->print(output);
    output << '\n';
    output << "Block Statements:\n";
    for(const auto& each : statements) {
        each->print(output);
    }
    output << '\n';
}

void LValue::print(std::ostream& output) const
{
    if(is_mutable) {
        output << "Variable: ";
    } else {
        output << "Constant: ";
    }

    output << name << '\n';
}

std::ostream& operator<<(std::ostream& output, const Function& function)
{
    output << "Function: " << function.name << '\n';
    output << "Parameters:\n";
    for(const auto *param : function.parameters) {
        param->print(output);
    }
    output << "\nStatements:\n";

    for(const auto& statement : function.statements) {
        statement->print(output);
    }
    return output;
}


std::ostream& operator<<(std::ostream& output, const Type& type)
{
    output << "Type: " << type.name << '\n';
    return output;
}
