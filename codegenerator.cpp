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
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Attributes.h>
#include <lld/Common/Driver.h>
#pragma GCC diagnostic pop

#include "ast.h"
#include <iostream>
#include <string>

BuiltinFunction BuiltinFunction::Print{"print"};

// Util functions
llvm::Value* truncate_to_bool(llvm::IRBuilder<>& ir_builder, llvm::Value* integer)
{
    return ir_builder.CreateTrunc(integer, llvm::Type::getInt1Ty(ir_builder.getContext()));
}


CodeGenerator::CodeGenerator(const char* source_filename,
                             const std::vector<Function*>& functions)
    : m_ir_builder(m_context), m_module(source_filename, m_context),
      m_functions(functions)
{
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    const std::string target_triple{llvm::sys::getDefaultTargetTriple()};

    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple, error);
    if(!error.empty()) {
        std::cerr << "Codegen error: " << error << "\n";
        exit(1);
    }
    const llvm::TargetOptions options;

    m_target_machine = target->createTargetMachine(
        target_triple,
        "generic",
        "",
        options, llvm::Reloc::PIC_, {}, {});

    m_module.setDataLayout(m_target_machine->createDataLayout());
    m_module.setTargetTriple(target_triple);
}

llvm::Value* StringLiteral::codegen(CodeGenerator& gen)
{
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
    return llvm::ConstantInt::get(gen.m_context,
                                  llvm::APInt(type()->bit_size(), value.str(), 10));
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

llvm::Value* BinaryExpression::codegen(CodeGenerator& gen)
{
    bool type_is_signed;
    if(left->type()->category() == TypeCategory::Range) {
        type_is_signed = static_cast<const RangeType*>(left->type())->is_signed();
    } else if(right->type()->category() == TypeCategory::Range) {
        type_is_signed = static_cast<const RangeType*>(right->type())->is_signed();
    } else {
        // TODO: add proper error printing code here
        assert(false);
    }
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
        if(type_is_signed)
            return gen.m_ir_builder.CreateSDiv(left_ir, right_ir, "sdivtmp");
        else
            return gen.m_ir_builder.CreateUDiv(left_ir, right_ir, "udivtmp");
    case TokenType::Op_Mult:
        return gen.m_ir_builder.CreateMul(left_ir, right_ir, "multmp");
    // TODO: Figure out IR instructions/usage of Op_Rem vs. Op_Mod
    case TokenType::Op_Mod:
        if(type_is_signed)
            return gen.m_ir_builder.CreateSRem(left_ir, right_ir, "smodtmp");
        else
            return gen.m_ir_builder.CreateURem(left_ir, right_ir, "umodtmp");
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
    case TokenType::Op_Lt:
        if(type_is_signed)
            return gen.m_ir_builder.CreateICmpSLT(left_ir, right_ir, "lttmp");
        else
            return gen.m_ir_builder.CreateICmpULT(left_ir, right_ir, "ulttmp");
    case TokenType::Op_Gt:
        if(type_is_signed)
            return gen.m_ir_builder.CreateICmpSGT(left_ir, right_ir, "gttmp");
        else
            return gen.m_ir_builder.CreateICmpUGT(left_ir, right_ir, "ugttmp");
    case TokenType::Op_Le:
        if(type_is_signed)
            return gen.m_ir_builder.CreateICmpSLE(left_ir, right_ir, "letmp");
        else
            return gen.m_ir_builder.CreateICmpULE(left_ir, right_ir, "uletmp");
    case TokenType::Op_Ge:
        if(type_is_signed)
            return gen.m_ir_builder.CreateICmpSGE(left_ir, right_ir, "getmp");
        else
            return gen.m_ir_builder.CreateICmpUGE(left_ir, right_ir, "ugetmp");
    case TokenType::Op_Left_Shift:
        return gen.m_ir_builder.CreateShl(left_ir, right_ir, "sltmp");
    case TokenType::Op_Right_Shift:
        // Fills in new bits with zeros
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
    llvm::CallInst* call_instr = gen.m_ir_builder.CreateCall(funct_to_call, args);
    // void function calls can't have a name (they don't return anything)
    if(type() != &Type::Void)
        call_instr->setName("call" + name);
    return call_instr;
}


void CodeGenerator::add_lvalue_init(llvm::Function* function, Statement* statement)
{
    auto* init = static_cast<Initialization*>(statement);
    LValue* lvalue = init->lvalue;

    auto prev_insert_point = m_ir_builder.saveIP();
    // All allocas need to be in the entry block
    // Allocas will be prepended to the entry block (so they all occur
    // before any are assigned values. This prevents LLVM passes from
    // creating new basic blocks and splitting up the allocas into different blocks)
    llvm::BasicBlock* entry = &function->getEntryBlock();
    m_ir_builder.SetInsertPoint(entry, entry->begin());

    llvm::AllocaInst* alloc = m_ir_builder.CreateAlloca(
        llvm::IntegerType::get(m_context, lvalue->type->range.bit_size), nullptr,
        lvalue->name.c_str());
    m_lvalues[lvalue] = alloc;
    m_ir_builder.restoreIP(prev_insert_point);

    if(init->expression != nullptr) {
        m_ir_builder.CreateStore(init->expression->codegen(*this), alloc);
    }
}

void CodeGenerator::in_statement(llvm::Function* curr_funct, Statement* statement)
{
    switch(statement->kind()) {
    case StatementKind::Basic: {
        auto* curr_statement = static_cast<BasicStatement*>(statement);
        curr_statement->expression->codegen(*this);
        break;
    }
    case StatementKind::Initialization:
        add_lvalue_init(curr_funct, statement);
        break;
    case StatementKind::Assignment:
        in_assignment(static_cast<Assignment*>(statement));
        break;
    case StatementKind::IfBlock:
        in_if_block(curr_funct, static_cast<IfBlock*>(statement));
        break;
    case StatementKind::While:
        in_while_loop(curr_funct, static_cast<WhileLoop*>(statement));
        break;
    case StatementKind::Block:
        // TODO: add support for anonymous blocks
        break;
    }
}

void CodeGenerator::in_assignment(Assignment* assgn)
{
    LValue* lvalue = assgn->lvalue;
    if(auto match = m_lvalues.find(lvalue); match != m_lvalues.end()) {
        llvm::AllocaInst* alloc = match->second;
        m_ir_builder.CreateStore(assgn->expression->codegen(*this), alloc);
    } else {
        std::cerr << "Codegen error: Could not find lvalue `"
                  << assgn->lvalue->name << "` in lvalue table\n";
        exit(1);
    }
}

void CodeGenerator::in_if_block(llvm::Function* curr_funct, IfBlock* ifblock,
                                llvm::BasicBlock* successor)
{
    llvm::Value* condition = ifblock->condition->codegen(*this);
    const auto cond_br_point = m_ir_builder.saveIP();

    // The if-block (jumped-to when condition is true)
    auto* if_true = llvm::BasicBlock::Create(m_context, "iftrue", curr_funct);
    m_ir_builder.SetInsertPoint(if_true);
    if(successor == nullptr) {
        successor = llvm::BasicBlock::Create(m_context, "successor", curr_funct);
    }
    in_block(curr_funct, ifblock, successor);

    // The linked block (i.e. else or else-if block); jumped to when false
    llvm::BasicBlock* if_false = successor;
    if(ifblock->else_or_else_if == nullptr) {
        // No more else-if or else-blocks; do nothing
    } else if(ifblock->else_or_else_if->kind() == StatementKind::IfBlock) {
        // Else-if block
        if_false = llvm::BasicBlock::Create(m_context, "iffalse", curr_funct);
        m_ir_builder.SetInsertPoint(if_false);
        in_if_block(curr_funct, static_cast<IfBlock*>(ifblock->else_or_else_if),
                    successor);
    } else if(ifblock->else_or_else_if->kind() == StatementKind::Block) {
        // Else block
        if_false = llvm::BasicBlock::Create(m_context, "iffalse", curr_funct);
        m_ir_builder.SetInsertPoint(if_false);
        in_block(curr_funct, ifblock->else_or_else_if, successor);
    }
    // Finally, insert the branch instruction right before the two branching blocks
    m_ir_builder.restoreIP(cond_br_point);
    m_ir_builder.CreateCondBr(condition, if_true, if_false);
    m_ir_builder.SetInsertPoint(successor);
}

// Currently used to generate else-blocks, but should work for anonymous blocks
// if they are added in the future
void CodeGenerator::in_block(llvm::Function* curr_funct,
                             Block* block, llvm::BasicBlock* successor)
{
    for(Statement* stmt : block->statements) {
        in_statement(curr_funct, stmt);
    }
    m_ir_builder.CreateBr(successor);
    m_ir_builder.SetInsertPoint(successor);
}

void CodeGenerator::in_while_loop(llvm::Function* curr_funct, WhileLoop* whileloop)
{
    auto* cond_block = llvm::BasicBlock::Create(m_context, "whileloophead", curr_funct);
    m_ir_builder.CreateBr(cond_block);
    m_ir_builder.SetInsertPoint(cond_block);

    llvm::Value* condition = whileloop->condition->codegen(*this);
    const auto cond_br_point = m_ir_builder.saveIP();

    auto* loop_body = llvm::BasicBlock::Create(m_context, "whileloopbody", curr_funct);
    auto* successor = llvm::BasicBlock::Create(m_context, "successor", curr_funct);
    m_ir_builder.SetInsertPoint(loop_body);
    // TODO: move as many loads as possible from loop body into the condition block;
    // this can lead to better code generation
    in_block(curr_funct, whileloop, cond_block);

    // Finally, insert the branch instruction right before the two branching blocks
    m_ir_builder.restoreIP(cond_br_point);
    m_ir_builder.CreateCondBr(condition, loop_body, successor);
    m_ir_builder.SetInsertPoint(successor);
}

void CodeGenerator::declare_builtin_functions()
{
    // int printf(const char*, ...);
    auto* StdInt = llvm::IntegerType::get(m_context, RangeType::Integer.bit_size());
    auto* CharPtr = llvm::PointerType::get(llvm::IntegerType::get(m_context, 8), 0);
    llvm::Type* params[] = { CharPtr };
    auto* printf_type = llvm::FunctionType::get(StdInt, params, true);
    auto* printf_funct = llvm::Function::Create(printf_type,
                                                llvm::GlobalValue::ExternalLinkage,
                                                "printf",
                                                m_module);
    printf_funct->setCallingConv(llvm::CallingConv::C);
    printf_funct->addFnAttr(llvm::Attribute::NoUnwind);
}

void CodeGenerator::declare_function_headers()
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
        curr_funct->addFnAttr(llvm::Attribute::NoUnwind);

        auto param_name_it = parameter_names.begin();
        for(auto& param : curr_funct->args()) {
            param.setName(*param_name_it);
            ++param_name_it;
        }

        parameter_types.clear();
        parameter_names.clear();
    }
}

void CodeGenerator::emit(const std::filesystem::path& object_file)
{
    std::error_code file_error;
    llvm::raw_fd_ostream output{object_file.string(), file_error};
    if(file_error) {
        std::cerr << "Codegen error: " << file_error.message() << "\n";
        return;
    }

    // We need this for some reason? Not really sure how to get around using it
    llvm::legacy::PassManager pass_manager;
    // TODO: add optimization passes
    m_target_machine->addPassesToEmitFile(pass_manager, output, nullptr,
                                          llvm::CodeGenFileType::CGFT_ObjectFile);
    pass_manager.run(m_module);
    output.flush();
}

void CodeGenerator::link(const std::filesystem::path& object_file,
                         const std::filesystem::path& exe_file)
{
#ifdef __APPLE__
    const char* args[] = { "lld", "-sdk_version", "10.14", "-o", exe_file.c_str(),
                           object_file.c_str(), "-lSystem" };
    if(!lld::mach_o::link(args, true, llvm::outs(), llvm::errs())) {
        std::cerr << "Linker failed\n";
        exit(1);
    }
#else
    std::cerr << "Note: linking not implemented for this platform, so"
        " no executable will be produced. Manually use linker to turn emitted"
        " object file into an executable\n";
#endif
}

void CodeGenerator::run()
{
    declare_function_headers();

    for(const Function *fcn : m_functions) {
        if(fcn->kind() != FunctionKind::Normal)
            // Builtin functions have no body
            continue;
        auto* function = static_cast<const BBFunction*>(fcn);
        llvm::Function* curr_funct = m_module.getFunction(function->name);
        // Next, create a block containing the body of the function
        auto* funct_body = llvm::BasicBlock::Create(m_context, "entry", curr_funct);
        m_ir_builder.SetInsertPoint(funct_body);
        for(auto *statement : function->statements) {
            in_statement(curr_funct, statement);
        }
        // All blocks must end in a ret instruction of some kind
        m_ir_builder.CreateRetVoid();
        llvm::verifyFunction(*curr_funct, &llvm::errs());
    }

    llvm::verifyModule(m_module, &llvm::errs());
    m_module.print(llvm::errs(), nullptr);

    std::filesystem::path object_file{m_module.getSourceFileName()};
    object_file.replace_extension(".o");
    object_file = object_file.filename();
    emit(object_file);
    link(object_file);
}
