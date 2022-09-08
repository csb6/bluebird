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
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
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
    case TypeKind::Literal: {
        auto* literal_type = static_cast<const LiteralType*>(ast_type);
        if(literal_type == &LiteralType::Char) {
            return mlir::IntegerType::get(&context, 8);
        } else if(literal_type == &LiteralType::Int) {
            // TODO: properly handle literal-only expressions (i.e. omit them from codegen)
            return mlir::IntegerType::get(&context, 64);
        } else {
            assert(false);
        }
    }
    default:
        assert(false);
    }
}

class IRExprVisitor : public ExprVisitor<IRExprVisitor> {
    mlir::OpBuilder& m_builder;
    std::unordered_map<const Assignable*, mlir::Value>& m_sse_vars;
    const std::unordered_map<const Function*, mlir::FuncOp>& m_mlir_functions;
public:
    IRExprVisitor(mlir::OpBuilder& builder,
                  std::unordered_map<const Assignable*, mlir::Value>& sse_vars,
                  const std::unordered_map<const struct Function*, mlir::FuncOp>& mlir_functions)
        : m_builder(builder), m_sse_vars(sse_vars), m_mlir_functions(mlir_functions) {}

    mlir::OpState on_visit(StringLiteral&) {}
    mlir::OpState on_visit(CharLiteral& literal)
    {
        auto mlir_type = to_mlir_type(*m_builder.getContext(), literal.type());
        return m_builder.create<mlir::ConstantOp>(
                    getLoc(m_builder, literal.line_num()),
                    mlir::IntegerAttr::get(mlir_type, literal.value));
    }
    mlir::OpState on_visit(IntLiteral& literal)
    {
        return m_builder.create<bluebirdIR::IntConstantOp>(
                    getLoc(m_builder, literal.line_num()),
                    literal.value,
                    to_mlir_type(*m_builder.getContext(), literal.type()));
    }
    mlir::OpState on_visit(BoolLiteral& literal)
    {
        return m_builder.create<mlir::ConstantOp>(
                    getLoc(m_builder, literal.line_num()),
                    mlir::BoolAttr::get(m_builder.getContext(), literal.value));
    }
    mlir::OpState on_visit(FloatLiteral&) {}
    mlir::OpState on_visit(VariableExpression& expr)
    {
        auto it = m_sse_vars.find(expr.variable);
        assert(it != m_sse_vars.end());
        return m_builder.create<mlir::memref::LoadOp>(getLoc(m_builder, expr.line_num()), it->second);
    }
    mlir::OpState on_visit(BinaryExpression& expr)
    {
        auto left = visitAndGetResult(expr.left.get());
        auto right = visitAndGetResult(expr.right.get());
        auto loc = getLoc(m_builder, expr.line_num());
        switch(expr.op) {
        case TokenType::Op_Plus:
            if(expr.type()->kind() == TypeKind::Range) {
                return m_builder.create<mlir::arith::AddIOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::arith::AddFOp>(loc, left, right);
            }
        case TokenType::Op_Minus:
            if(expr.type()->kind() == TypeKind::Range) {
                return m_builder.create<mlir::arith::SubIOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::arith::SubFOp>(loc, left, right);
            }
        case TokenType::Op_Div:
            return m_builder.create<bluebirdIR::DivideOp>(loc, left, right);
        case TokenType::Op_Mult:
            if(expr.type()->kind() == TypeKind::Range) {
                return m_builder.create<mlir::arith::MulIOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::arith::MulFOp>(loc, left, right);
            }
        case TokenType::Op_And:
            return m_builder.create<mlir::arith::AndIOp>(loc, left, right);
        case TokenType::Op_Or:
            return m_builder.create<mlir::arith::OrIOp>(loc, left, right);
        case TokenType::Op_Xor:
            return m_builder.create<mlir::arith::XOrIOp>(loc, left, right);
        case TokenType::Op_Eq:
            // TODO: do float version
            return m_builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, left, right);
        case TokenType::Op_Ne:
            // TODO: do float version
            return m_builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, left, right);
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
    mlir::OpState on_visit(UnaryExpression& expr)
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
    mlir::OpState on_visit(FunctionCall& function_call)
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
    mlir::OpState on_visit(IndexOp&) {}
    mlir::OpState on_visit(InitList&) {}

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
    std::unordered_map<const Assignable*, mlir::Value>& m_sse_vars;
    const std::unordered_map<const struct Function*, mlir::FuncOp>& m_mlir_functions;
    IRExprVisitor m_expr_visitor;
    mlir::FuncOp m_curr_function;
public:
    IRStmtVisitor(mlir::OpBuilder& builder,
                  std::unordered_map<const Assignable*, mlir::Value>& sse_vars,
                  const std::unordered_map<const Function*, mlir::FuncOp>& mlir_functions)
        : m_builder(builder), m_sse_vars(sse_vars), m_mlir_functions(mlir_functions),
          m_expr_visitor(m_builder, m_sse_vars, m_mlir_functions) {}

    void set_function_and_alloca_args(const BBFunction& user_function, mlir::FuncOp mlir_function)
    {
        m_curr_function = mlir_function;
        auto& entry_block = getEntryBlock(mlir_function);
        m_builder.setInsertionPointToStart(&entry_block);

        // Arguments are passed as SSA values, so create allocas for them so they can be modified in
        // the function body
        auto argLoc = mlir_function.getLoc();
        llvm::SmallVector<mlir::Value, 4> param_allocas;
        for(auto& ast_param : user_function.parameters) {
            auto memref_type = mlir::MemRefType::get(
                        {}, to_mlir_type(*m_builder.getContext(), ast_param->type));
            auto alloca = m_builder.create<mlir::memref::AllocaOp>(argLoc, std::move(memref_type));
            m_sse_vars.insert_or_assign(ast_param.get(), alloca);
            param_allocas.push_back(std::move(alloca));
        }
        assert(param_allocas.size() == mlir_function.getArguments().size());
        for(auto[alloca, mlir_arg] : llvm::zip(param_allocas, mlir_function.getArguments())) {
            m_builder.create<mlir::memref::StoreOp>(argLoc, mlir_arg, std::move(alloca));
        }
    }

    void on_visit(BasicStatement& stmt)
    {
        m_expr_visitor.visit(*stmt.expression.get());
    }
    void on_visit(Initialization& var_decl)
    {
        auto memref_type = mlir::MemRefType::get(
                    {}, to_mlir_type(*m_builder.getContext(), var_decl.variable->type));
        auto loc = getLoc(m_builder, var_decl.line_num());

        // Put allocas at start of function's entry block
        auto oldInsertionPoint = m_builder.saveInsertionPoint();
        auto& entry_block = getEntryBlock(m_curr_function);
        m_builder.setInsertionPointToStart(&entry_block);
        auto alloca = m_builder.create<mlir::memref::AllocaOp>(loc, std::move(memref_type));
        m_builder.restoreInsertionPoint(oldInsertionPoint);
        if(var_decl.expression != nullptr) {
            auto value = m_expr_visitor.visitAndGetResult(var_decl.expression.get());
            m_builder.create<mlir::memref::StoreOp>(loc, value, alloca);
        }
        m_sse_vars.insert_or_assign(var_decl.variable.get(), std::move(alloca));
    }
    void on_visit(Assignment& asgmt)
    {
        auto alloca = m_sse_vars.find(asgmt.assignable);
        assert(alloca != m_sse_vars.end());
        auto value = m_expr_visitor.visitAndGetResult(asgmt.expression.get());
        m_builder.create<mlir::memref::StoreOp>(value.getLoc(), value, alloca->second);
    }
    void on_visit(IfBlock&) {}
    void on_visit(Block&) {}
    void on_visit(WhileLoop&) {}
    void on_visit(ReturnStatement& stmt)
    {
        if(stmt.expression == nullptr) {
            // In void function
            m_builder.create<mlir::ReturnOp>(getLoc(m_builder, stmt.line_num()), llvm::None);
        } else {
            // In non-void function
            auto return_value = m_expr_visitor.visitAndGetResult(stmt.expression.get());
            m_builder.create<mlir::ReturnOp>(return_value.getLoc(), return_value);
        }
    }
};

namespace bluebirdIR {

Builder::Builder(std::vector<Magnum::Pointer<Function>>& functions,
        std::vector<Magnum::Pointer<Type>>& types,
        std::vector<Magnum::Pointer<Initialization>>& global_vars,
        std::vector<Magnum::Pointer<IndexedVariable>>& index_vars)
    : m_module(mlir::ModuleOp::create(mlir::OpBuilder(&m_context).getUnknownLoc())),
      m_builder(m_module.getRegion()),
      m_functions(functions), m_types(types), m_global_vars(global_vars), m_index_vars(index_vars)
{
    m_context.loadDialect<BluebirdIRDialect, mlir::StandardOpsDialect, mlir::memref::MemRefDialect>();
}

void Builder::run()
{
    llvm::SmallVector<mlir::Type, 4> param_types;
    // First, create all of the functions (with empty bodies). This ensures that
    // all function calls will resolve to an existing mlir::FuncOp
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

    // Next, generate the function bodies and parameters
    IRStmtVisitor stmt_visitor(m_builder, m_sse_vars, m_mlir_functions);
    for(auto& function : m_functions) {
        if(function->kind() == FunctionKind::Normal) {
            auto* user_function = static_cast<BBFunction*>(function.get());
            auto& mlir_function = m_mlir_functions[user_function];
            stmt_visitor.set_function_and_alloca_args(*user_function, mlir_function);

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
