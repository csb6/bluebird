#include "cleanup.h"
#include "ast.h"
#include "error.h"

Cleanup::Cleanup(std::vector<Magnum::Pointer<Function>>& functions,
                 std::vector<Magnum::Pointer<Initialization>>& global_vars)
    : m_functions(functions), m_global_vars(global_vars)
{}

static
void fold_unary_constants(Magnum::Pointer<Expression>& expr_location_out,
                          UnaryExpression* unary_expr)
{
    auto replace_with_literal =
        [&](Expression* literal) {
            auto* released = unary_expr->right.release();
            assert(released == literal);
            expr_location_out.reset(literal);
        };

    switch(unary_expr->right->kind()) {
    case ExprKind::IntLiteral: {
        auto* r_int = static_cast<IntLiteral*>(unary_expr->right.get());
        switch(unary_expr->op) {
        case TokenType::Op_Minus:
            // TODO: use r_int->value.ones_complement(); for certain unsigned integer types
            r_int->value.negate();
            break;
        default:
            raise_error_expected("unary expression with operator that works on "
                                 "integer literals", unary_expr);
        }
        replace_with_literal(r_int);
        break;
    }
    case ExprKind::BoolLiteral: {
        auto* r_bool = static_cast<BoolLiteral*>(unary_expr->right.get());
        if(unary_expr->op == TokenType::Op_Not) {
            r_bool->value = !r_bool->value;
        } else {
            raise_error_expected("unary expression with logical operator "
                                 "(e.g. `not`)", unary_expr);
        }
        replace_with_literal(r_bool);
        break;
    }
    default:
        // No constant folding needed
        break;
    }
}

static
void fold_binary_constants(Magnum::Pointer<Expression>& expr_location_out,
                           BinaryExpression* bin_expr)
{
    auto replace_with_literal =
        [&](Expression* literal) {
            auto* released = bin_expr->left.release();
            assert(released == literal);
            expr_location_out.reset(literal);
        };

    const auto left_kind = bin_expr->left->kind();
    const auto right_kind = bin_expr->right->kind();
    if(left_kind == ExprKind::IntLiteral && right_kind == ExprKind::IntLiteral) {
        // Fold into a single literal (uses arbitrary-precision arithmetic)
        auto* l_int = static_cast<IntLiteral*>(bin_expr->left.get());
        auto* r_int = static_cast<IntLiteral*>(bin_expr->right.get());
        switch(bin_expr->op) {
        case TokenType::Op_Plus:
            l_int->value += r_int->value;
            break;
        case TokenType::Op_Minus:
            l_int->value -= r_int->value;
            break;
        case TokenType::Op_Mult:
            l_int->value *= r_int->value;
            break;
        case TokenType::Op_Div:
            l_int->value /= r_int->value;
            break;
        case TokenType::Op_Mod:
            l_int->value.mod(r_int->value);
            break;
        case TokenType::Op_Rem:
            l_int->value.rem(r_int->value);
            break;
        // TODO: Add support for shift operators
        case TokenType::Op_Thru:
        case TokenType::Op_Upto:
            // Can't do folds here, need to preserve left/right sides for a range
            return;
        default:
            if(is_bool_op(bin_expr->op)) {
                raise_error_expected("binary expression with an operator that works "
                                     "on integer literals", bin_expr);
            }
            return;
        }
        replace_with_literal(l_int);
    } else if(left_kind == ExprKind::BoolLiteral && right_kind == ExprKind::BoolLiteral) {
        // Fold into a single literal
        auto* l_bool = static_cast<BoolLiteral*>(bin_expr->left.get());
        auto* r_bool = static_cast<BoolLiteral*>(bin_expr->right.get());
        switch(bin_expr->op) {
        case TokenType::Op_And:
            l_bool->value &= r_bool->value;
            break;
        case TokenType::Op_Or:
            l_bool->value |= r_bool->value;
            break;
        case TokenType::Op_Xor:
            l_bool->value ^= r_bool->value;
            break;
        default:
            raise_error_expected("binary expression with logical operator", bin_expr);
        }
        replace_with_literal(l_bool);
    }
}

static
void fold_constants(Magnum::Pointer<Expression>& child_location)
{
    child_location->cleanup();
    if(child_location->kind() == ExprKind::Binary) {
        fold_binary_constants(child_location,
                              static_cast<BinaryExpression*>(child_location.get()));
    } else if(child_location->kind() == ExprKind::Unary) {
        fold_unary_constants(child_location,
                             static_cast<UnaryExpression*>(child_location.get()));
    }
}

void UnaryExpression::cleanup()
{
    fold_constants(right);
}

void BinaryExpression::cleanup()
{
    fold_constants(left);
    fold_constants(right);
}

void FunctionCall::cleanup()
{
    for(auto& arg : arguments) {
        fold_constants(arg);
    }
}

void IndexOp::cleanup()
{
    // Base expression will not consist of anything that can be simplified down
    // to a single literal (because it is some sort of variable usage), so no need
    // to try and fold it; instead, just call cleanup() so any of its subexpressions
    // are folded if possible.
    base_expr->cleanup();

    fold_constants(index_expr);
}

void InitList::cleanup()
{
    for(auto& val : values) {
        fold_constants(val);
    }
}

void BasicStatement::cleanup()
{
    fold_constants(expression);
}

void Initialization::cleanup()
{
    if(expression != nullptr) {
        fold_constants(expression);
    }
}

void Assignment::cleanup()
{
    fold_constants(expression);
}

void Block::cleanup()
{
    for(auto& stmt : statements) {
        stmt->cleanup();
    }
}

void IfBlock::cleanup()
{
    fold_constants(condition);
    Block::cleanup();
    if(else_or_else_if != nullptr) {
        else_or_else_if->cleanup();
    }
}

void WhileLoop::cleanup()
{
    fold_constants(condition);
    Block::cleanup();
}

void ReturnStatement::cleanup()
{
    if(expression != nullptr) {
        fold_constants(expression);
    }
}

void Cleanup::run()
{
    for(Magnum::Pointer<Initialization>& var : m_global_vars) {
        var->cleanup();
    }
    for(auto& funct : m_functions) {
        if(funct->kind() == FunctionKind::Normal) {
            auto* normal_funct = static_cast<BBFunction*>(funct.get());
            normal_funct->body.cleanup();
        }
    }
}
