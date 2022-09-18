/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2021-2022  Cole Blakley

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
#include "constanteval.h"
#include "ast.h"
#include "error.h"

static
void fold_unary_constants(Magnum::Pointer<Expression>& expr_location_out,
                          UnaryExpression* unary_expr)
{
    auto replace_with_literal =
        [&](Expression* literal) {
            #ifndef NDEBUG
                auto* released = unary_expr->right.release();
                assert(released == literal);
            #else
                unary_expr->right.release();
            #endif
            expr_location_out.reset(literal);
        };

    switch(unary_expr->right->kind()) {
    case ExprKind::IntLiteral: {
        auto* r_int = as<IntLiteral>(unary_expr->right.get());
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
        auto* r_bool = as<BoolLiteral>(unary_expr->right.get());
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
            #ifndef NDEBUG
                auto* released = bin_expr->left.release();
                assert(released == literal);
            #else
                bin_expr->left.release();
            #endif
            expr_location_out.reset(literal);
        };

    const auto left_kind = bin_expr->left->kind();
    const auto right_kind = bin_expr->right->kind();
    if(left_kind == ExprKind::IntLiteral && right_kind == ExprKind::IntLiteral) {
        // Fold into a single literal (uses arbitrary-precision arithmetic)
        auto* l_int = as<IntLiteral>(bin_expr->left.get());
        auto* r_int = as<IntLiteral>(bin_expr->right.get());
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
        case TokenType::Op_Eq:
        case TokenType::Op_Ne:
        case TokenType::Op_Lt:
        case TokenType::Op_Gt:
        case TokenType::Op_Le:
        case TokenType::Op_Ge:
            // Can't do folds here, need to preserve left/right sides for a range or comparison
            return;
        default:
            raise_error_expected("binary expression with an operator that evaluates "
                                 "to integer literals", bin_expr);
            return;
        }
        replace_with_literal(l_int);
    } else if(left_kind == ExprKind::BoolLiteral && right_kind == ExprKind::BoolLiteral) {
        // Fold into a single literal
        auto* l_bool = as<BoolLiteral>(bin_expr->left.get());
        auto* r_bool = as<BoolLiteral>(bin_expr->right.get());
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

void fold_constants(Magnum::Pointer<Expression>& expr)
{
    switch(expr->kind()) {
    case ExprKind::Binary:
        fold_binary_constants(expr, as<BinaryExpression>(expr.get()));
        break;
    case ExprKind::Unary:
        fold_unary_constants(expr, as<UnaryExpression>(expr.get()));
        break;
    default:
        // Ignore other kinds of expressions; they can't be folded
        break;
    }
}
