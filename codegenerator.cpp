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
#include <string>

// Util functions
llvm::Value* truncate_to_bool(llvm::IRBuilder<>& ir_builder, llvm::Value* integer)
{
    return ir_builder.CreateTrunc(integer, llvm::Type::getInt1Ty(ir_builder.getContext()));
}


CodeGenerator::CodeGenerator(const std::vector<Function*>& functions)
    : m_ir_builder(m_context), m_module("main module", m_context),
      m_functions(functions)
{}

llvm::Value* StringLiteral::codegen(CodeGenerator& gen)
{
    // TODO: make sure this doesn't mess-up the insertion point for IR instructions
    return gen.m_ir_builder.CreateGlobalStringPtr(value);
}

llvm::Value* CharLiteral::codegen(CodeGenerator& gen)
{
    // Create a signed `char` type
    return llvm::ConstantInt::get(gen.m_context,
                                  llvm::APInt(sizeof(value), value, true));
}

llvm::Value* IntLiteral::codegen(CodeGenerator& gen)
{
    // TODO: determine signed-ness; right now, assume signed
    return llvm::ConstantInt::get(gen.m_context,
                                  llvm::APInt(bit_size, value.str(), 10));
}

llvm::Value* FloatLiteral::codegen(CodeGenerator& gen)
{
    return llvm::ConstantFP::get(gen.m_context, llvm::APFloat(value));
}

llvm::Value* LValueExpression::codegen(CodeGenerator& gen)
{
    llvm::AllocaInst* src_lvalue = gen.m_lvalues[lvalue];
    if(src_lvalue == nullptr) {
        std::cerr << "Codegen error: Could not find lvalue referenced by:\n";
        print(std::cerr);
        exit(1);
    }
    return gen.m_ir_builder.CreateLoad(src_lvalue, name);
}

llvm::Value* UnaryExpression::codegen(CodeGenerator& gen)
{
    llvm::Value* operand = right->codegen(gen);

    switch(op) {
    case TokenType::Op_Not:
        operand = gen.m_ir_builder.CreateNot(operand, "nottmp");
        return truncate_to_bool(gen.m_ir_builder, operand);
    case TokenType::Op_Bit_Not:
        return gen.m_ir_builder.CreateNot(operand, "bitnottmp");
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

llvm::Value* BinaryExpression::codegen(CodeGenerator& gen)
{
    set_int_literal_bit_size(left.get(), right.get());
    llvm::Value* left_ir = left->codegen(gen);
    llvm::Value* right_ir = right->codegen(gen);

    // TODO: use float/user-defined versions of all these operators depending
    //   on their type
    switch(op) {
    case TokenType::Op_Plus:
        return gen.m_ir_builder.CreateAdd(left_ir, right_ir, "addtmp");
    case TokenType::Op_Minus:
        return gen.m_ir_builder.CreateSub(left_ir, right_ir, "subtmp");
    case TokenType::Op_Div:
        // TODO: use different division depending on signed-ness.
        // For now, assume signed
        return gen.m_ir_builder.CreateSDiv(left_ir, right_ir, "sdivtmp");
    case TokenType::Op_Mult:
        return gen.m_ir_builder.CreateMul(left_ir, right_ir, "multmp");
    case TokenType::Op_Mod:
        // TODO: use different division depending on signed-ness.
        // For now, assume signed
        return gen.m_ir_builder.CreateSRem(left_ir, right_ir, "modtmp");
    // TODO: implement short-circuiting for AND and OR
    case TokenType::Op_And:
        left_ir = gen.m_ir_builder.CreateAnd(left_ir, right_ir, "andtmp");
        return truncate_to_bool(gen.m_ir_builder, left_ir);
    case TokenType::Op_Or:
        left_ir = gen.m_ir_builder.CreateOr(left_ir, right_ir, "ortmp");
        return truncate_to_bool(gen.m_ir_builder, left_ir);
    // TODO: use different compare functions for floats, user-defined ops, etc.
    case TokenType::Op_Eq:
        return gen.m_ir_builder.CreateICmpEQ(left_ir, right_ir, "eqtmp");
    case TokenType::Op_Ne:
        return gen.m_ir_builder.CreateICmpNE(left_ir, right_ir, "netmp");
    // TODO: vary compare function if signed or not
    // For now, assume signed
    case TokenType::Op_Lt:
        return gen.m_ir_builder.CreateICmpSLT(left_ir, right_ir, "lttmp");
    case TokenType::Op_Gt:
        return gen.m_ir_builder.CreateICmpSGT(left_ir, right_ir, "gttmp");
    case TokenType::Op_Le:
        return gen.m_ir_builder.CreateICmpSLE(left_ir, right_ir, "letmp");
    case TokenType::Op_Ge:
        return gen.m_ir_builder.CreateICmpSGE(left_ir, right_ir, "getmp");
    case TokenType::Op_Left_Shift:
        return gen.m_ir_builder.CreateShl(left_ir, right_ir, "sltmp");
    case TokenType::Op_Right_Shift:
        // Fills in opened bits with zeros
        return gen.m_ir_builder.CreateLShr(left_ir, right_ir, "srtmp");
    case TokenType::Op_Bit_And:
        return gen.m_ir_builder.CreateAnd(left_ir, right_ir, "bitandtmp");
    case TokenType::Op_Bit_Or:
        return gen.m_ir_builder.CreateOr(left_ir, right_ir, "bitortmp");
    case TokenType::Op_Bit_Xor:
        return gen.m_ir_builder.CreateXor(left_ir, right_ir, "bitxortmp");
    default:
        assert(false && "Unknown binary operator");
    }
}

llvm::Value* FunctionCall::codegen(CodeGenerator& gen)
{
    llvm::Function* funct_to_call = gen.m_module.getFunction(name);
    if(funct_to_call == nullptr) {
        std::cerr << "Codegen error: Could not find function `"
                  << name << "`\n";
        exit(1);
    }
    std::vector<llvm::Value*> args;
    args.reserve(arguments.size());
    for(auto& arg : arguments) {
        args.push_back(arg->codegen(gen));
    }
    return gen.m_ir_builder.CreateCall(funct_to_call, args, "call" + name);
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
                                                  m_module);
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
                curr_statement->expression->codegen(*this);
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

    m_module.print(llvm::errs(), nullptr);
}
