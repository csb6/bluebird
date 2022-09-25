#include "builder.h"
#include "dialect.h"
#include "types.h"
#include "visitor.h"
#include "ast.h"
#include "error.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/Verifier.h>
#pragma GCC diagnostic pop
#include <iostream>
#include <cstdint>

static mlir::Location to_loc(mlir::OpBuilder& builder, unsigned int line_num)
{
    return mlir::FileLineColLoc::get(builder.getContext(), "", line_num, 1);
}

static mlir::Block* getEntryBlock(mlir::FuncOp function)
{
    assert(!function.getBlocks().empty());
    auto& entry_block = function.getBlocks().front();
    assert(entry_block.isEntryBlock());
    return &entry_block;
}

static mlir::Type to_mlir_type(mlir::MLIRContext& context, const Type* ast_type)
{
    if(ast_type == &Type::Void) {
        return mlir::NoneType::get(&context);
    }

    switch(ast_type->kind()) {
    case TypeKind::IntRange:
    case TypeKind::Boolean:
        return mlir::IntegerType::get(&context, ast_type->bit_size());
    case TypeKind::FloatRange:
        return mlir::Float32Type::get(&context);
    case TypeKind::Ptr:
        return mlir::MemRefType::get({}, to_mlir_type(context, as<PtrType>(ast_type)->inner_type));
    case TypeKind::Literal: {
        auto* literal_type = as<LiteralType>(ast_type);
        if(literal_type == &LiteralType::Char) {
            return mlir::IntegerType::get(&context, 8);
        } else if(literal_type == &LiteralType::Int) {
            return mlir::IntegerType::get(&context, 64);
        } else if(literal_type == &LiteralType::Float) {
            return mlir::FloatType::getF32(&context);
        } else {
            assert(false);
        }
    }
    default:
        assert(false);
    }
}

static constexpr std::uint8_t cmp_f_instr_lookup[] = {
    /*Op_Eq*/ (std::uint8_t)mlir::arith::CmpFPredicate::OEQ,
    /*Op_Ne*/ (std::uint8_t)mlir::arith::CmpFPredicate::ONE,
    /*Op_Lt*/ (std::uint8_t)mlir::arith::CmpFPredicate::OLT,
    /*Op_Gt*/ (std::uint8_t)mlir::arith::CmpFPredicate::OGT,
    /*Op_Le*/ (std::uint8_t)mlir::arith::CmpFPredicate::OLE,
    /*Op_Ge*/ (std::uint8_t)mlir::arith::CmpFPredicate::OGE
};

static
mlir::arith::CmpFPredicate get_cmp_f_pred(TokenType op_type)
{
    return (mlir::arith::CmpFPredicate)cmp_f_instr_lookup[(char)op_type - (char)TokenType::Op_Eq];
}

static constexpr struct {
    std::uint8_t s_cmp;
    std::uint8_t u_cmp;
} cmp_i_instr_lookup[] = {
    /*Op_Eq*/ { (std::uint8_t)mlir::arith::CmpIPredicate::eq, (std::uint8_t)mlir::arith::CmpIPredicate::eq },
    /*Op_Ne*/ { (std::uint8_t)mlir::arith::CmpIPredicate::ne, (std::uint8_t)mlir::arith::CmpIPredicate::ne },
    /*Op_Lt*/ { (std::uint8_t)mlir::arith::CmpIPredicate::slt, (std::uint8_t)mlir::arith::CmpIPredicate::ult },
    /*Op_Gt*/ { (std::uint8_t)mlir::arith::CmpIPredicate::sgt, (std::uint8_t)mlir::arith::CmpIPredicate::ugt },
    /*Op_Le*/ { (std::uint8_t)mlir::arith::CmpIPredicate::sle, (std::uint8_t)mlir::arith::CmpIPredicate::ule },
    /*Op_Ge*/ { (std::uint8_t)mlir::arith::CmpIPredicate::sge, (std::uint8_t)mlir::arith::CmpIPredicate::uge }
};
static_assert((std::uint8_t)TokenType::Op_Eq + 5 == (std::uint8_t)TokenType::Op_Ge);

static
mlir::arith::CmpIPredicate get_cmp_i_pred(TokenType op_type, bool is_signed)
{
    if(is_signed) {
        return (mlir::arith::CmpIPredicate)cmp_i_instr_lookup[(char)op_type - (char)TokenType::Op_Eq].s_cmp;
    } else {
        return (mlir::arith::CmpIPredicate)cmp_i_instr_lookup[(char)op_type - (char)TokenType::Op_Eq].u_cmp;
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
        return m_builder.create<bluebirdIR::CharConstantOp>(
                    to_loc(m_builder, literal.line_num()),
                    literal.value);
    }

    mlir::OpState on_visit(IntLiteral& literal)
    {
        return m_builder.create<bluebirdIR::IntConstantOp>(
                    to_loc(m_builder, literal.line_num()),
                    literal.value,
                    to_mlir_type(*m_builder.getContext(), literal.type()));
    }

    mlir::OpState on_visit(BoolLiteral& literal)
    {
        return m_builder.create<bluebirdIR::BoolConstantOp>(
                    to_loc(m_builder, literal.line_num()),
                    literal.value);
    }

    mlir::OpState on_visit(FloatLiteral& literal)
    {
        return m_builder.create<bluebirdIR::FloatConstantOp>(
                    to_loc(m_builder, literal.line_num()),
                    literal.value);
    }

    mlir::OpState on_visit(VariableExpression& expr)
    {
        auto it = m_sse_vars.find(expr.variable);
        assert(it != m_sse_vars.end());
        return m_builder.create<mlir::memref::LoadOp>(to_loc(m_builder, expr.line_num()), it->second);
    }

    mlir::OpState on_visit(BinaryExpression& expr)
    {
        auto left = visitAndGetResult(expr.left.get());
        auto right = visitAndGetResult(expr.right.get());
        auto loc = to_loc(m_builder, expr.line_num());
        assert(left.getType() == right.getType());
        bool is_float_operand = expr.left->type()->kind() == TypeKind::FloatRange;
        bool is_signed = false;
        if(expr.left->type()->kind() == TypeKind::IntRange) {
            is_signed = as<IntRangeType>(expr.left->type())->is_signed();
        } else if(expr.right->type()->kind() == TypeKind::IntRange) {
            is_signed = as<IntRangeType>(expr.right->type())->is_signed();
        }

        switch(expr.op) {
        case TokenType::Op_Plus:
            if(is_float_operand) {
                return m_builder.create<mlir::arith::AddFOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::arith::AddIOp>(loc, left, right);
            }
        case TokenType::Op_Minus:
            if(is_float_operand) {
                return m_builder.create<mlir::arith::SubFOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::arith::SubIOp>(loc, left, right);
            }
        case TokenType::Op_Div:
            if(is_float_operand) {
                return m_builder.create<mlir::arith::DivFOp>(loc, left, right);
            } else if(is_signed) {
                return m_builder.create<mlir::arith::DivSIOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::arith::DivUIOp>(loc, left, right);
            }
        case TokenType::Op_Mult:
            if(is_float_operand) {
                return m_builder.create<mlir::arith::MulFOp>(loc, left, right);
            } else {
                return m_builder.create<mlir::arith::MulIOp>(loc, left, right);
            }
        case TokenType::Op_And:
            return m_builder.create<mlir::arith::AndIOp>(loc, left, right);
        case TokenType::Op_Or:
            return m_builder.create<mlir::arith::OrIOp>(loc, left, right);
        case TokenType::Op_Xor:
            return m_builder.create<mlir::arith::XOrIOp>(loc, left, right);
        case TokenType::Op_Eq:
        case TokenType::Op_Ne:
        case TokenType::Op_Lt:
        case TokenType::Op_Gt:
        case TokenType::Op_Le:
        case TokenType::Op_Ge:
            if(is_float_operand) {
                return m_builder.create<mlir::arith::CmpFOp>(loc, get_cmp_f_pred(expr.op), left, right);
            } else {
                assert(expr.type()->kind() == TypeKind::IntRange || expr.type()->kind() == TypeKind::Boolean);
                return m_builder.create<mlir::arith::CmpIOp>(loc, get_cmp_i_pred(expr.op, is_signed), left, right);
            }
        default:
            Error().put("MLIR: Unhandled operator").quote(expr.op).raise();
        }
    }

    mlir::OpState on_visit(UnaryExpression& expr)
    {
        auto right = visitAndGetResult(expr.right.get());
        auto loc = to_loc(m_builder, expr.line_num());
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

        return m_builder.create<mlir::CallOp>(to_loc(m_builder, function_call.line_num()),
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
        m_builder.setInsertionPointToStart(getEntryBlock(mlir_function));

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
        m_expr_visitor.visit(*stmt.expression);
    }

    void on_visit(Initialization& var_decl)
    {
        auto memref_type = mlir::MemRefType::get(
                    {}, to_mlir_type(*m_builder.getContext(), var_decl.variable->type));
        auto loc = to_loc(m_builder, var_decl.line_num());
        // Put allocas at start of function's entry block
        auto oldInsertionPoint = m_builder.saveInsertionPoint();
        m_builder.setInsertionPointToStart(getEntryBlock(m_curr_function));
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
        m_builder.create<mlir::memref::StoreOp>(to_loc(m_builder, asgmt.line_num()), value, alloca->second);
    }

    void on_visit(IfBlock& if_else_stmt)
    {
        auto condition = m_expr_visitor.visitAndGetResult(if_else_stmt.condition.get());
        auto loc = to_loc(m_builder, if_else_stmt.line_num());
        bool has_else_or_else_if = if_else_stmt.else_or_else_if != nullptr;

        auto if_else_block = m_builder.create<mlir::scf::IfOp>(loc, condition, has_else_or_else_if);
        auto old_insert_point = m_builder.saveInsertionPoint();
        m_builder.setInsertionPointToStart(if_else_block.thenBlock());
        for(auto& stmt : if_else_stmt.statements) {
            visit(*stmt);
        }

        if(has_else_or_else_if) {
            auto* else_block = if_else_block.elseBlock();
            assert(else_block != nullptr);
            m_builder.setInsertionPointToStart(else_block);
            if(if_else_stmt.else_or_else_if->kind() == StmtKind::IfBlock) {
                // Else-if block (create a new if-block inside the else block)
                visit(*if_else_stmt.else_or_else_if);
            } else {
                // Else block (just put the statements directly into the block)
                for(auto& stmt : if_else_stmt.else_or_else_if->statements) {
                    visit(*stmt);
                }
            }
        }
        m_builder.restoreInsertionPoint(old_insert_point);
    }

    void on_visit(Block& block)
    {
        auto loc = to_loc(m_builder, block.line_num());
        auto region = m_builder.create<mlir::scf::ExecuteRegionOp>(loc, m_builder.getNoneType());
        auto old_insert_point = m_builder.saveInsertionPoint();
        m_builder.setInsertionPoint(region);
        for(auto& stmt : block.statements) {
            visit(*stmt);
        }
        m_builder.restoreInsertionPoint(old_insert_point);
    }

    void on_visit(WhileLoop& while_stmt)
    {
        auto while_loc = to_loc(m_builder, while_stmt.line_num());
        auto while_loop = m_builder.create<mlir::scf::WhileOp>(while_loc, mlir::TypeRange{}, mlir::ValueRange{});
        auto old_insert_point = m_builder.saveInsertionPoint();

        auto& condition_block = while_loop.getBefore().emplaceBlock();
        m_builder.setInsertionPointToStart(&condition_block);
        auto condition_expr = m_expr_visitor.visitAndGetResult(while_stmt.condition.get());
        auto condition_expr_loc = to_loc(m_builder, while_stmt.condition->line_num());
        m_builder.create<mlir::scf::ConditionOp>(condition_expr_loc, condition_expr, mlir::ValueRange{});

        auto& loop_body = while_loop.getAfter().emplaceBlock();
        m_builder.setInsertionPointToStart(&loop_body);
        for(auto& stmt : while_stmt.statements) {
            visit(*stmt);
        }
        m_builder.create<mlir::scf::YieldOp>(while_loc);
        m_builder.restoreInsertionPoint(old_insert_point);
    }

    void on_visit(ReturnStatement& stmt)
    {
        auto loc = to_loc(m_builder, stmt.line_num());
        if(stmt.expression == nullptr) {
            // In void function
            m_builder.create<mlir::ReturnOp>(loc);
        } else {
            // In non-void function
            auto return_value = m_expr_visitor.visitAndGetResult(stmt.expression.get());
            m_builder.create<mlir::ReturnOp>(loc, return_value);
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
    m_context.loadDialect<BluebirdIRDialect, mlir::StandardOpsDialect, mlir::arith::ArithmeticDialect,
                          mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
}

void Builder::run()
{
    llvm::SmallVector<mlir::Type, 4> param_types;
    // First, create all of the functions (with empty bodies). This ensures that
    // all function calls will resolve to an existing mlir::FuncOp
    for(auto& function : m_functions) {
        if(function->kind() == FunctionKind::Normal) {
            auto* user_function = as<BBFunction>(function.get());
            for(auto& param : user_function->parameters) {
                param_types.push_back(to_mlir_type(m_context, param->type));
            }
            mlir::TypeRange result_type = to_mlir_type(m_context, user_function->return_type);
            if(result_type.front().isa<mlir::NoneType>()) {
                result_type = {};
            }
            auto func_type = m_builder.getFunctionType(param_types, result_type);
            auto mlir_funct = m_builder.create<mlir::FuncOp>(to_loc(m_builder, user_function->line_num()),
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
            auto* user_function = as<BBFunction>(function.get());
            auto& mlir_function = m_mlir_functions[user_function];
            stmt_visitor.set_function_and_alloca_args(*user_function, mlir_function);

            for(auto& statement : user_function->body.statements) {
                stmt_visitor.visit(*statement.get());
            }
            if(user_function->return_type == &Type::Void) {
                m_builder.setInsertionPointToEnd(getEntryBlock(mlir_function));
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
