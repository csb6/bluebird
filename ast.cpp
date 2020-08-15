#include "ast.h"

bool Range::contains(long value) const
{
    return value >= lower_bound && value <= upper_bound;
}


void LValueExpression::print(std::ostream& output) const
{
    output << "LValueExpr: " << name << '\n';
}

UnaryExpression::UnaryExpression(TokenType oper, Expression* r)
    : op(oper), right(r)
{}

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
    for(const auto& param : function.parameters) {
        output << param->name << ' ';
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
