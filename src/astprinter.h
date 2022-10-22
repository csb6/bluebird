#ifndef ASTPRINTER_H
#define ASTPRINTER_H
#include <visitor.h>
#include <iosfwd>

class ASTExprPrinter : public ExprVisitor<ASTExprPrinter, VisitorKind::Const> {
    std::ostream& m_output;
public:
    explicit
    ASTExprPrinter(std::ostream& output) : m_output(output) {}

    void on_visit(const struct StringLiteral&);
    void on_visit(const struct CharLiteral&);
    void on_visit(const struct IntLiteral&);
    void on_visit(const struct BoolLiteral&);
    void on_visit(const struct FloatLiteral&);
    void on_visit(const struct VariableExpression&);
    void on_visit(const struct BinaryExpression&);
    void on_visit(const struct UnaryExpression&);
    void on_visit(const struct FunctionCall&);
    void on_visit(const struct IndexedExpr&);
    void on_visit(const struct InitList&);
};

class ASTStmtPrinter : public StmtVisitor<ASTStmtPrinter, VisitorKind::Const> {
    ASTExprPrinter m_expr_visitor;
    std::ostream& m_output;
    unsigned int m_indent;
public:
    explicit
    ASTStmtPrinter(std::ostream& output, unsigned int indent = 0)
        : m_expr_visitor(output), m_output(output), m_indent(indent) {}

    void on_visit(const struct BasicStatement&);
    void on_visit(const struct Initialization&);
    void on_visit(const struct Assignment&);
    void on_visit(const struct IfBlock&);
    void on_visit(const struct Block&);
    void on_visit(const struct WhileLoop&);
    void on_visit(const struct ReturnStatement&);
};

std::ostream& operator<<(std::ostream&, const struct Expression&);
std::ostream& operator<<(std::ostream&, const struct Statement&);
std::ostream& operator<<(std::ostream&, const struct IntRange&);
std::ostream& operator<<(std::ostream&, const struct FloatRange&);
std::ostream& operator<<(std::ostream&, const struct Assignable&);
std::ostream& operator<<(std::ostream&, const struct Function&);
#endif // ASTPRINTER_H
