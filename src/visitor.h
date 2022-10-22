#ifndef BLUEBIRD_VISITOR_H
#define BLUEBIRD_VISITOR_H
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
#include "ast.h"
#include <type_traits>

enum class VisitorKind {
    Const, Mutable
};

template<VisitorKind K, typename T>
using VisitorQual = std::conditional_t<K == VisitorKind::Mutable, T, const T>;

template<typename DerivedT, VisitorKind Kind = VisitorKind::Mutable>
struct ExprVisitor {
    using ExprType = VisitorQual<Kind, Expression>;

    auto visit(ExprType& expr)
    {
        #define CALL_VISIT(Type) \
            (static_cast<DerivedT*>(this)->on_visit(static_cast<VisitorQual<Kind, Type>&>(expr)))

        switch(expr.kind()) {
        case ExprKind::StringLiteral:
            return CALL_VISIT(StringLiteral);
        case ExprKind::CharLiteral:
            return CALL_VISIT(CharLiteral);
        case ExprKind::IntLiteral:
            return CALL_VISIT(IntLiteral);
        case ExprKind::BoolLiteral:
            return CALL_VISIT(BoolLiteral);
        case ExprKind::FloatLiteral:
            return CALL_VISIT(FloatLiteral);
        case ExprKind::Variable:
            return CALL_VISIT(VariableExpression);
        case ExprKind::Binary:
            return CALL_VISIT(BinaryExpression);
        case ExprKind::Unary:
            return CALL_VISIT(UnaryExpression);
        case ExprKind::FunctionCall:
            return CALL_VISIT(FunctionCall);
        case ExprKind::IndexOp:
            return CALL_VISIT(IndexOp);
        case ExprKind::InitList:
            return CALL_VISIT(InitList);
        }
        #undef CALL_VISIT
    }
};

template<typename DerivedT, VisitorKind Kind = VisitorKind::Mutable>
struct StmtVisitor {
    using StmtType = VisitorQual<Kind, Statement>;

    auto visit(StmtType& stmt)
    {
        #define CALL_VISIT(Type) \
            static_cast<DerivedT*>(this)->on_visit(static_cast<VisitorQual<Kind, Type>&>(stmt))

        switch(stmt.kind()) {
        case StmtKind::Basic:
            return CALL_VISIT(BasicStatement);
        case StmtKind::Initialization:
            return CALL_VISIT(Initialization);
        case StmtKind::Assignment:
            return CALL_VISIT(Assignment);
        case StmtKind::IfBlock:
            return CALL_VISIT(IfBlock);
        case StmtKind::Block:
            return CALL_VISIT(Block);
        case StmtKind::While:
            return CALL_VISIT(WhileLoop);
        case StmtKind::Return:
            return CALL_VISIT(ReturnStatement);
        }
        #undef CALL_VISIT
    }
};

#endif
