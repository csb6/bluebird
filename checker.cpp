#include "checker.h"
#include "ast.h"
#include <iostream>

Checker::Checker(const std::vector<Function*>& functions,
                 const std::vector<RangeType*>& types)
    : m_functions(functions), m_types(types)
{}

void Checker::check_types(const Statement* statement) const
{
    switch(statement->type()) {
    case StatementType::Basic: {
        auto* actual = static_cast<const BasicStatement*>(statement);
        check_types(statement, actual->expression.get());
        break;
    }
    case StatementType::Initialization:
        break;
    case StatementType::IfBlock:
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
    switch(expression->expr_type()) {
    case ExpressionType::Binary: {
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
            std::cerr << "\n  Operator: " << actual->op << "\n  Right: ";
            actual->right->print(std::cerr);
            std::cerr << '\n';
            exit(1);
        }
        break;
    }
    case ExpressionType::FunctionCall:
        break;
    case ExpressionType::Unary:
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
