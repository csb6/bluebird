#include "builder.h"
#include "dialect.h"
#include "types.h"
#include "visitor.h"
#include "ast.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Verifier.h>
#pragma GCC diagnostic pop
#include <iostream>

static mlir::Location getLoc(mlir::OpBuilder& builder, unsigned int line_num)
{
    return mlir::FileLineColLoc::get(builder.getContext(), "", line_num, 1);
}

static mlir::Type to_mlir_type(mlir::MLIRContext& context, const Type* ast_type)
{
    if(ast_type == &Type::Void) {
        return mlir::NoneType::get(&context);
    }

    switch(ast_type->kind()) {
    case TypeKind::Range:
    case TypeKind::Boolean:
        return mlir::IntegerType::get(&context, ast_type->bit_size());
    case TypeKind::Ptr: {
        auto* ptr_type = static_cast<const PtrLikeType*>(ast_type);
        return mlir::MemRefType::get({}, to_mlir_type(context, ptr_type->inner_type));
    }
    default:
        assert(false);
    }
}

class IRExprVisitor : public ExprVisitor<IRExprVisitor> {
    mlir::OpBuilder& m_builder;
    std::unordered_map<const Variable*, mlir::Value>& m_sse_vars;
    const std::unordered_map<const Function*, mlir::FuncOp>& m_mlir_functions;
public:
    IRExprVisitor(mlir::OpBuilder& builder,
                  std::unordered_map<const struct Variable*, mlir::Value>& sse_vars,
                  const std::unordered_map<const struct Function*, mlir::FuncOp>& mlir_functions)
        : m_builder(builder), m_sse_vars(sse_vars), m_mlir_functions(mlir_functions) {}

    mlir::OpState visit_impl(StringLiteral&) {}
    mlir::OpState visit_impl(CharLiteral& literal)
    {
        return m_builder.create<mlir::ConstantOp>(
                    getLoc(m_builder, literal.line_num()),
                    bluebirdIR::CharLiteralAttr::get(m_builder.getContext(), literal.value));
    }
    mlir::OpState visit_impl(IntLiteral& literal)
    {
        return m_builder.create<bluebirdIR::IntConstantOp>(
                    getLoc(m_builder, literal.line_num()),
                    literal.value,
                    to_mlir_type(*m_builder.getContext(), literal.type()));
    }
    mlir::OpState visit_impl(BoolLiteral& literal)
    {
        return m_builder.create<mlir::ConstantOp>(
                    getLoc(m_builder, literal.line_num()),
                    mlir::BoolAttr::get(m_builder.getContext(), literal.value));
    }
    mlir::OpState visit_impl(FloatLiteral&) {}
    mlir::OpState visit_impl(VariableExpression& expr)
    {
        auto it = m_sse_vars.find(expr.variable);
        assert(it != m_sse_vars.end());
        return m_builder.create<mlir::memref::LoadOp>(getLoc(m_builder, expr.line_num()), it->second);
    }
    mlir::OpState visit_impl(BinaryExpression& expr)
    {
        auto left = visitAndGetResult(expr.left.get());
        auto right = visitAndGetResult(expr.right.get());
        auto loc = getLoc(m_builder, expr.line_num());
        switch(expr.op) {
        case TokenType::Op_Plus:
            if(expr.type()->kind() == TypeKind::Range) {
                return m_builder.create<mlir::AddIOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::AddFOp>(loc, left, right);
            }
        case TokenType::Op_Minus:
            if(expr.type()->kind() == TypeKind::Range) {
                return m_builder.create<mlir::SubIOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::SubFOp>(loc, left, right);
            }
        case TokenType::Op_Div:
            return m_builder.create<bluebirdIR::DivideOp>(loc, left, right);
        case TokenType::Op_Mult:
            if(expr.type()->kind() == TypeKind::Range) {
                return m_builder.create<mlir::MulIOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::MulFOp>(loc, left, right);
            }
        case TokenType::Op_And:
            return m_builder.create<mlir::AndOp>(loc, left, right);
        case TokenType::Op_Or:
            return m_builder.create<mlir::OrOp>(loc, left, right);
        case TokenType::Op_Xor:
            return m_builder.create<mlir::XOrOp>(loc, left, right);
        case TokenType::Op_Eq:
            // TODO: do float version
            return m_builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::eq, left, right);
        case TokenType::Op_Ne:
            // TODO: do float version
            return m_builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::ne, left, right);
        case TokenType::Op_Lt:
            //return m_builder.create<mlir::CmpIOp>(loc, left, right);
        case TokenType::Op_Gt:
            //return m_builder.create<mlir::CmpIOp>(loc, left, right);
        case TokenType::Op_Le:
            //return m_builder.create<mlir::CmpIOp>(loc, left, right);
        case TokenType::Op_Ge:
            //return m_builder.create<mlir::CmpIOp>(loc, left, right);
        default:
            assert(false);
        }
    }
    mlir::OpState visit_impl(UnaryExpression& expr)
    {
        auto right = visitAndGetResult(expr.right.get());
        auto loc = getLoc(m_builder, expr.line_num());
        switch(expr.op) {
        case TokenType::Op_Minus:
            return m_builder.create<bluebirdIR::NegateOp>(loc, right);
        case TokenType::Op_Not:
            return m_builder.create<bluebirdIR::NotOp>(loc, right);
        default:
            assert(false);
        }
    }
    mlir::OpState visit_impl(FunctionCall& function_call)
    {
        llvm::SmallVector<mlir::Value, 4> arguments;
        for(auto& arg : function_call.arguments) {
            arguments.push_back(visitAndGetResult(arg.get()));
        }
        auto function_def = m_mlir_functions.find(function_call.definition);
        assert(function_def != m_mlir_functions.end());

        return m_builder.create<mlir::CallOp>(getLoc(m_builder, function_call.line_num()),
                                              function_def->second,
                                              mlir::ValueRange(arguments));
    }
    mlir::OpState visit_impl(IndexOp&) {}
    mlir::OpState visit_impl(InitList&) {}

    mlir::Value visitAndGetResult(Expression* expr)
    {
        // All expressions return a single result, so this is safe to do.
        return visit(*expr)->getResult(0);
    }
};

static mlir::Block& getEntryBlock(mlir::FuncOp function)
{
    assert(!function.getBlocks().empty());
    auto& entry_block = function.getBlocks().front();
    assert(entry_block.isEntryBlock());
    return entry_block;
}

class IRStmtVisitor : public StmtVisitor<IRStmtVisitor> {
    mlir::OpBuilder& m_builder;
    std::unordered_map<const struct Variable*, mlir::Value>& m_sse_vars;
    const std::unordered_map<const struct Function*, mlir::FuncOp>& m_mlir_functions;
    IRExprVisitor m_expr_visitor;
    mlir::FuncOp m_curr_function;
public:
    IRStmtVisitor(mlir::OpBuilder& builder,
                  std::unordered_map<const Variable*, mlir::Value>& sse_vars,
                  const std::unordered_map<const Function*, mlir::FuncOp>& mlir_functions)
        : m_builder(builder), m_sse_vars(sse_vars), m_mlir_functions(mlir_functions),
          m_expr_visitor(m_builder, m_sse_vars, m_mlir_functions) {}

    void set_function(mlir::FuncOp function)
    {
        m_curr_function = function;
        auto& entry_block = getEntryBlock(function);
        m_builder.setInsertionPointToStart(&entry_block);
    }

    void visit_impl(BasicStatement& stmt)
    {
        m_expr_visitor.visit(*stmt.expression.get());
    }
    void visit_impl(Initialization& var_decl)
    {
        auto memref_type = mlir::MemRefType::get(
                    {}, to_mlir_type(*m_builder.getContext(), var_decl.variable->type));

        // Put allocas at start of function's entry block
        auto oldInsertionPoint = m_builder.saveInsertionPoint();
        auto& entry_block = getEntryBlock(m_curr_function);
        m_builder.setInsertionPointToStart(&entry_block);
        auto alloca = m_builder.create<mlir::memref::AllocaOp>(getLoc(m_builder, var_decl.line_num()),
                                                               std::move(memref_type));
        m_builder.restoreInsertionPoint(oldInsertionPoint);
        if(var_decl.expression != nullptr) {
            //auto value = m_expr_visitor.visitAndGetResult(var_decl.expression.get());
        }
        m_sse_vars.insert_or_assign(var_decl.variable.get(), std::move(alloca));
    }
    void visit_impl(Assignment&) {}
    void visit_impl(IfBlock&) {}
    void visit_impl(Block&) {}
    void visit_impl(WhileLoop&) {}
    void visit_impl(ReturnStatement&) {}
};

namespace bluebirdIR {

Builder::Builder(std::vector<Magnum::Pointer<struct Function>>& functions,
        std::vector<Magnum::Pointer<struct Type>>& types,
        std::vector<Magnum::Pointer<struct Initialization>>& global_vars,
        std::vector<Magnum::Pointer<struct IndexedVariable>>& index_vars)
    : m_module(mlir::ModuleOp::create(mlir::OpBuilder(&m_context).getUnknownLoc())),
      m_builder(m_module.getRegion()),
      m_functions(functions), m_types(types), m_global_vars(global_vars), m_index_vars(index_vars)
{
    m_context.loadDialect<BluebirdIRDialect, mlir::StandardOpsDialect, mlir::memref::MemRefDialect>();
}

void Builder::run()
{
    llvm::SmallVector<mlir::Type, 4> param_types;
    for(auto& function : m_functions) {
        if(function->kind() == FunctionKind::Normal) {
            auto* user_function = static_cast<BBFunction*>(function.get());
            for(auto& param : user_function->parameters) {
                param_types.push_back(to_mlir_type(m_context, param->type));
            }
            mlir::TypeRange result_type = to_mlir_type(m_context, user_function->return_type);
            if(result_type.front().isa<mlir::NoneType>()) {
                result_type = {};
            }
            auto func_type = m_builder.getFunctionType(param_types, result_type);
            auto mlir_funct = m_builder.create<mlir::FuncOp>(getLoc(m_builder, user_function->line_num()),
                                                             user_function->name,
                                                             func_type);
            mlir_funct.addEntryBlock();
            m_mlir_functions.insert_or_assign(user_function, mlir_funct);
            param_types.clear();
        }
    }

    IRStmtVisitor stmt_visitor(m_builder, m_sse_vars, m_mlir_functions);
    for(auto& function : m_functions) {
        if(function->kind() == FunctionKind::Normal) {
            auto* user_function = static_cast<BBFunction*>(function.get());
            auto& mlir_function = m_mlir_functions[user_function];
            stmt_visitor.set_function(mlir_function);
            for(auto& statement : user_function->body.statements) {
                stmt_visitor.visit(*statement.get());
            }
            auto& entry_block = getEntryBlock(mlir_function);
            if(user_function->return_type == &Type::Void) {
                m_builder.setInsertionPointToEnd(&entry_block);
                m_builder.create<mlir::ReturnOp>(m_builder.getUnknownLoc(), mlir::None);
            }
        }
    }
    m_module.dump();
    auto result = mlir::verify(m_module.getOperation());
    if(result.failed()) {
        std::cerr << "Failed to verify module\n";
    }
}

};
