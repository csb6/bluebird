#include <astprinter.h>
#include <ast.h>
#include <token.h>
#include <ostream>

static
std::ostream& indent_by(std::ostream& output, unsigned int indent)
{
    for(unsigned int i = 0; i < indent; ++i) {
        output << ' ';
    }
    return output;
}

void ASTExprPrinter::on_visit(const StringLiteral& lit)
{
    m_output << "\"";
    print_unescape(lit.value, m_output);
    m_output << lit.value << "\"";
}

void ASTExprPrinter::on_visit(const CharLiteral& lit)
{
    m_output << "'";
    print_unescape(lit.value, m_output);
    m_output << lit.value << "'";
}

void ASTExprPrinter::on_visit(const IntLiteral& lit)
{
    m_output << lit.value;
}

void ASTExprPrinter::on_visit(const BoolLiteral& lit)
{
    m_output << lit.value;
}

void ASTExprPrinter::on_visit(const FloatLiteral& lit)
{
    m_output << lit.value;
}

void ASTExprPrinter::on_visit(const VariableExpression& var_expr)
{
    m_output << var_expr.variable->name;
}

void ASTExprPrinter::on_visit(const BinaryExpression& bin_expr)
{
    m_output << "(";
    visit(*bin_expr.left);
    m_output << " " << bin_expr.op << " ";
    visit(*bin_expr.right);
    m_output << ")";
}

void ASTExprPrinter::on_visit(const UnaryExpression& unary_expr)
{
    m_output << unary_expr.op << " ";
    visit(*unary_expr.right);
}

void ASTExprPrinter::on_visit(const FunctionCall& func_call)
{
    m_output << func_call.name() << "(";
    for(const auto& arg : func_call.arguments) {
        visit(*arg);
        m_output << ", ";
    }
    m_output << ")";
}

void ASTExprPrinter::on_visit(const IndexedExpr& index_op)
{
    m_output << "(";
    visit(*index_op.base_expr);
    m_output << ")[";
    visit(*index_op.index_expr);
    m_output << "]";
}

void ASTExprPrinter::on_visit(const InitList& init_list)
{
    m_output << "{ ";
    for(const auto& each : init_list.values) {
        visit(*each);
        m_output << ", ";
    }
    m_output << "}";
}


void ASTStmtPrinter::on_visit(const BasicStatement& stmt)
{
    indent_by(m_output, m_indent);
    m_expr_visitor.visit(*stmt.expression);
    m_output << ";\n";
}

void ASTStmtPrinter::on_visit(const Initialization& init)
{
    indent_by(m_output, m_indent);
    m_output << "let " <<  init.variable->name << ": " << init.variable->type->name;
    if(init.expression != nullptr) {
        m_output << " := ";
        m_expr_visitor.visit(*init.expression);
    }
    m_output << ";\n";
}

void ASTStmtPrinter::on_visit(const Assignment& asgmt)
{
    indent_by(m_output, m_indent);
    m_output << *asgmt.assignable << " := ";
    m_expr_visitor.visit(*asgmt.expression);
    m_output << ";\n";
}

// TODO: fix weird indentation when printing these
void ASTStmtPrinter::on_visit(const IfBlock& if_blocks)
{
    indent_by(m_output, m_indent);
    m_output << "if ";
    m_expr_visitor.visit(*if_blocks.condition);
    m_output << " do\n";
    ++m_indent;
    for(const auto& stmt : if_blocks.statements) {
        visit(*stmt);
    }
    if(if_blocks.else_or_else_if != nullptr) {
        indent_by(m_output, m_indent - 1) << "else ";
        visit(*if_blocks.else_or_else_if);
    }
    --m_indent;
}

void ASTStmtPrinter::on_visit(const Block& block)
{
    for(const auto& stmt : block.statements) {
        visit(*stmt);
    }
}

void ASTStmtPrinter::on_visit(const WhileLoop& while_loop)
{
    indent_by(m_output, m_indent) << "while ";
    m_expr_visitor.visit(*while_loop.condition);
    m_output << " do\n";
    ++m_indent;
    for(const auto& stmt : while_loop.statements) {
        visit(*stmt);
    }
    --m_indent;
    indent_by(m_output, m_indent);
    m_output << "end\n";
}

void ASTStmtPrinter::on_visit(const ReturnStatement& stmt)
{
    indent_by(m_output, m_indent) << "return";
    if(stmt.expression != nullptr) {
        m_output << " ";
        m_expr_visitor.visit(*stmt.expression);
    }
    m_output << ";\n";
}


std::ostream& operator<<(std::ostream& output, const IntRange& range)
{
    output << "(" << range.lower_bound << ", " << range.upper_bound << ")";
    return output;
}

std::ostream& operator<<(std::ostream& output, const FloatRange& range)
{
    output << "(" << range.lower_bound << ", " << range.upper_bound << ")";
    return output;
}

std::ostream& operator<<(std::ostream& output, const Expression& expr)
{
    ASTExprPrinter{output}.visit(expr);
    return output;
}

std::ostream& operator<<(std::ostream& output, const Statement& stmt)
{
    ASTStmtPrinter{output}.visit(stmt);
    return output;
}

std::ostream& operator<<(std::ostream& output, const Assignable& assignable)
{
    switch(assignable.kind()) {
    case AssignableKind::Variable:
        output << as<Variable>(assignable).name;
        break;
    case AssignableKind::Deref:
        output << "*(" << as<DerefLValue>(assignable).ptr_var << ")";
        break;
    case AssignableKind::Indexed: {
        const auto& index_assign = *as<IndexedVariable>(assignable).indexed_expr;
        ASTExprPrinter expr_printer{output};
        output << "(";
        expr_printer.visit(*index_assign.base_expr);
        output << ")[";
        expr_printer.visit(*index_assign.index_expr);
        output << "]";
        break;
    }
    }
    return output;
}

std::ostream& operator<<(std::ostream& output, const Function& function)
{
    output << "function " << function.name << "(";
    for(const auto& param : function.parameters) {
        output << param->name << ": " << param->type->name << ", ";
    }
    output << ") ";

    if(function.kind() == FunctionKind::Normal) {
        const auto& fcn = as<BBFunction>(function);
        ASTStmtPrinter stmt_printer{output, 1};
        output << "is\n";
        for(const auto& stmt : fcn.body.statements) {
            stmt_printer.visit(*stmt);
        }
        output << "end\n";
    } else {
        output << "is external;\n";
    }
    return output;
}
