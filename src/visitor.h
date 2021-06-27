#ifndef BLUEBIRD_VISITOR_H
#define BLUEBIRD_VISITOR_H
#include "ast.h"

template<typename DerivedT>
struct ExprVisitor {
    auto visit(Expression& expr)
    {
        #define CALL_VISIT(Type) \
            static_cast<DerivedT*>(this)->visit_impl(static_cast<Type&>(expr))

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

template<typename DerivedT>
struct StmtVisitor {
    auto visit(Statement& stmt)
    {
        #define CALL_VISIT(Type) \
            static_cast<DerivedT*>(this)->visit_impl(static_cast<Type&>(stmt))

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
