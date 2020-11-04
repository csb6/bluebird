/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020  Cole Blakley

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
#include "codegenerator.h"
// Internal LLVM code has lots of warnings that we don't
// care about, so ignore them
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Instructions.h>
#pragma GCC diagnostic pop

#include "ast.h"
#include <iostream>

// Util functions
llvm::Value* truncate_to_bool(llvm::IRBuilder<>& ir_builder, llvm::Value* integer)
{
    return ir_builder.CreateTrunc(integer, llvm::Type::getInt1Ty(ir_builder.getContext()));
}


CodeGenerator::CodeGenerator(const std::vector<Function*>& functions)
    : m_ir_builder(m_context), 
      m_curr_module(new llvm::Module("main module", m_context)),
      m_functions(functions)
{}

llvm::Value* CodeGenerator::in_expression(Expression* expression)
{
    switch(expression->expr_type()) {
    case ExpressionType::StringLiteral:
        return in_string_literal(expression);
    case ExpressionType::CharLiteral:
        return in_char_literal(expression);
    case ExpressionType::IntLiteral:
        return in_int_literal(expression);
    case ExpressionType::FloatLiteral:
        return in_float_literal(expression);
    case ExpressionType::LValue:
        return in_lvalue_expression(expression);
    case ExpressionType::Unary:
        return in_unary_expression(expression);
    case ExpressionType::Binary:
        return in_binary_expression(expression);
    case ExpressionType::FunctionCall:
        return in_function_call(expression);
    }
}

llvm::Value* CodeGenerator::in_string_literal(const Expression* expression)
{
    auto* literal = static_cast<const StringLiteral*>(expression);
    // TODO: make sure this doesn't mess-up the insertion point for IR instructions
    return m_ir_builder.CreateGlobalStringPtr(literal->value);
}

llvm::Value* CodeGenerator::in_char_literal(const Expression* expression)
{
    auto* literal = static_cast<const CharLiteral*>(expression);
    const auto value = literal->value;
    // Create a signed `char` type
    return llvm::ConstantInt::get(m_context,
                                  llvm::APInt(sizeof(value), value, true));
}

llvm::Value* CodeGenerator::in_int_literal(const Expression* expression)
{
    auto* literal = static_cast<const IntLiteral*>(expression);
    const auto value = literal->value;
    // TODO: determine signed-ness; right now, assume signed
    return llvm::ConstantInt::get(m_context, llvm::APInt(literal->bit_size,
                                                         value.str(), 10));
}

llvm::Value* CodeGenerator::in_float_literal(const Expression* expression)
{
    auto* literal = static_cast<const FloatLiteral*>(expression);
    return llvm::ConstantFP::get(m_context, llvm::APFloat(literal->value));
}

llvm::Value* CodeGenerator::in_lvalue_expression(const Expression* expression)
{
    auto* lvalue_expr = static_cast<const LValueExpression*>(expression);
    llvm::AllocaInst* src_lvalue = m_lvalues[lvalue_expr->lvalue];
    if(src_lvalue == nullptr) {
        std::cerr << "Codegen error: Could not find lvalue referenced by:\n";
        lvalue_expr->print(std::cerr);
        exit(1);
    }
    return m_ir_builder.CreateLoad(src_lvalue, lvalue_expr->name);
}

llvm::Value* CodeGenerator::in_unary_expression(Expression* expression)
{
    auto* expr = static_cast<UnaryExpression*>(expression);
    llvm::Value* operand = in_expression(expr->right.get());

    switch(expr->op) {
    case TokenType::Op_Not:
        operand = m_ir_builder.CreateNot(operand, "nottmp");
        return truncate_to_bool(m_ir_builder, operand);
    case TokenType::Op_Bit_Not:
        return m_ir_builder.CreateNot(operand, "bitnottmp");
    default:
        assert(false && "Unknown unary operator");
    }
}

void set_int_literal_bit_size(Expression* l, Expression* r)
{
    const ExpressionType l_type = l->expr_type();
    const ExpressionType r_type = r->expr_type();
    if(l_type != ExpressionType::IntLiteral && r_type != ExpressionType::IntLiteral) {
        return;
    } else if(l_type == ExpressionType::IntLiteral) {
        // literal op non-literal
        auto* literal = static_cast<IntLiteral*>(l);
        literal->bit_size = r->type()->bit_size();
    } else if(r_type == ExpressionType::IntLiteral) {
        // non-literal op literal
        auto* literal = static_cast<IntLiteral*>(r);
        literal->bit_size = l->type()->bit_size();
    } else {
        // literal op literal (this case should have already been removed)
        assert(false);
    }
}

llvm::Value* CodeGenerator::in_binary_expression(Expression* expression)
{
    auto* expr = static_cast<BinaryExpression*>(expression);
    set_int_literal_bit_size(expr->left.get(), expr->right.get());
    llvm::Value* left = in_expression(expr->left.get());
    llvm::Value* right = in_expression(expr->right.get());

    // TODO: use float/user-defined versions of all these operators depending
    //   on their type
    switch(expr->op) {
    case TokenType::Op_Plus:
        return m_ir_builder.CreateAdd(left, right, "addtmp");
    case TokenType::Op_Minus:
        return m_ir_builder.CreateSub(left, right, "subtmp");
    case TokenType::Op_Div:
        // TODO: use different division depending on signed-ness.
        // For now, assume signed
        return m_ir_builder.CreateSDiv(left, right, "sdivtmp");
    case TokenType::Op_Mult:
        return m_ir_builder.CreateMul(left, right, "multmp");
    case TokenType::Op_Mod:
        // TODO: use different division depending on signed-ness.
        // For now, assume signed
        return m_ir_builder.CreateSRem(left, right, "modtmp");
    // TODO: implement short-circuiting for AND and OR
    case TokenType::Op_And:
        left = m_ir_builder.CreateAnd(left, right, "andtmp");
        return truncate_to_bool(m_ir_builder, left);
    case TokenType::Op_Or:
        left = m_ir_builder.CreateOr(left, right, "ortmp");
        return truncate_to_bool(m_ir_builder, left);
    // TODO: use different compare functions for floats, user-defined ops, etc.
    case TokenType::Op_Eq:
        return m_ir_builder.CreateICmpEQ(left, right, "eqtmp");
    case TokenType::Op_Ne:
        return m_ir_builder.CreateICmpNE(left, right, "netmp");
    // TODO: vary compare function if signed or not
    // For now, assume signed
    case TokenType::Op_Lt:
        return m_ir_builder.CreateICmpSLT(left, right, "lttmp");
    case TokenType::Op_Gt:
        return m_ir_builder.CreateICmpSGT(left, right, "gttmp");
    case TokenType::Op_Le:
        return m_ir_builder.CreateICmpSLE(left, right, "letmp");
    case TokenType::Op_Ge:
        return m_ir_builder.CreateICmpSGE(left, right, "getmp");
    case TokenType::Op_Left_Shift:
        return m_ir_builder.CreateShl(left, right, "sltmp");
    case TokenType::Op_Right_Shift:
        // Fills in opened bits with zeros
        return m_ir_builder.CreateLShr(left, right, "srtmp");
    case TokenType::Op_Bit_And:
        return m_ir_builder.CreateAnd(left, right, "bitandtmp");
    case TokenType::Op_Bit_Or:
        return m_ir_builder.CreateOr(left, right, "bitortmp");
    case TokenType::Op_Bit_Xor:
        return m_ir_builder.CreateXor(left, right, "bitxortmp");
    default:
        assert(false && "Unknown binary operator");
    }
}

llvm::Value* CodeGenerator::in_function_call(Expression* expression)
{
    auto* funct_call = static_cast<FunctionCall*>(expression);
    llvm::Function* funct_to_call = m_curr_module->getFunction(funct_call->name);
    if(funct_to_call == nullptr) {
        std::cerr << "Codegen error: Could not find function `"
                  << funct_call->name << "`\n";
        exit(1);
    }
    std::vector<llvm::Value*> args;
    args.reserve(funct_call->arguments.size());
    for(auto& arg : funct_call->arguments) {
        args.push_back(in_expression(arg.get()));
    }
    return m_ir_builder.CreateCall(funct_to_call, args, "call" + funct_call->name);
}


void CodeGenerator::add_lvalue_init(llvm::Function* function, Statement* statement)
{
    auto* init = static_cast<const Initialization*>(statement);
    llvm::IRBuilder<> builder(&function->getEntryBlock(),
                              function->getEntryBlock().begin());
    auto* lvalue = init->lvalue;

    llvm::AllocaInst* alloc = builder.CreateAlloca(
        llvm::IntegerType::get(m_context, lvalue->type->range.bit_size), 0,
        lvalue->name.c_str());
    m_lvalues[lvalue] = alloc;
}


void CodeGenerator::run()
{
    // Reused between iterations to reduce allocations
    std::vector<llvm::Type*> parameter_types;
    std::vector<llvm::StringRef> parameter_names;
    for(const Function *function : m_functions) {
        // First, create a function declaration
        parameter_types.reserve(function->parameters.size());
        parameter_names.reserve(function->parameters.size());

        for(const auto *param : function->parameters) {
            parameter_types.push_back(
                llvm::Type::getIntNTy(m_context, param->type->range.bit_size));
            parameter_names.push_back(param->name);
        }

        // TODO: add support for return types other than void
        auto* funct_type = llvm::FunctionType::get(llvm::Type::getVoidTy(m_context),
                                                   parameter_types, false);
        // TODO: add more fine-grained support for Linkage
        auto* curr_funct = llvm::Function::Create(funct_type,
                                                  llvm::Function::ExternalLinkage,
                                                  function->name,
                                                  m_curr_module.get());
        assert(curr_funct->getParent() != nullptr);
        
        auto param_name_it = parameter_names.begin();
        for(auto& param : curr_funct->args()) {
            param.setName(*param_name_it);
            ++param_name_it;
        }

        parameter_types.clear();
        parameter_names.clear();

        // Next, create a block containing the body of the function
        auto* funct_body = llvm::BasicBlock::Create(m_context, "body", curr_funct);
        m_ir_builder.SetInsertPoint(funct_body);
        for(auto *statement : function->statements) {
            switch(statement->type()) {
            case StatementType::Basic: {
                auto* curr_statement = static_cast<BasicStatement*>(statement);
                in_expression(curr_statement->expression.get());
                break;
            }
            case StatementType::Initialization:
                add_lvalue_init(curr_funct, statement);
                break;
            case StatementType::IfBlock:
                // TODO: add support for if-blocks
                break;
            }
        }
    }

    m_curr_module->print(llvm::errs(), nullptr);
}
