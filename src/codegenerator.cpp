/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2021  Cole Blakley

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
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include "optimizer.h"
#include "objectgenerator.h"
#pragma GCC diagnostic pop

#include "ast.h"
#include "error.h"
#include <string>

DebugGenerator::DebugGenerator(bool is_active, llvm::Module& module,
                               const char* source_filename)
    : m_dbg_builder(module), m_is_active(is_active)
{
    if(is_active) {
        m_dbg_unit = m_dbg_builder.createCompileUnit(
            llvm::dwarf::DW_LANG_C, m_dbg_builder.createFile(source_filename, "."),
            "Bluebird Compiler", false, "", 0);
        m_file = m_dbg_builder.createFile(m_dbg_unit->getFilename(),
                                          m_dbg_unit->getDirectory());
        // dbg_unit is root scope
        m_scopes.push_back(m_dbg_unit);
    }
}

llvm::DIType* DebugGenerator::to_dbg_type(const Type* ast_type)
{
    // See http://www.dwarfstd.org/doc/DWARF5.pdf, page 227
    // TODO: add support for determining LLVM debug type for other AST types
    unsigned encoding;
    auto bit_size = ast_type->bit_size();
    switch(ast_type->kind()) {
    case TypeKind::Range: {
        auto* type = static_cast<const RangeType*>(ast_type);
        if(type->range.is_signed)
            encoding = llvm::dwarf::DW_ATE_signed;
        else
            encoding = llvm::dwarf::DW_ATE_unsigned;
        break;
    }
    case TypeKind::Boolean:
        encoding = llvm::dwarf::DW_ATE_boolean;
        // While the correct bit_size is 1, lldb crashes if bit_size < 8;
        // lldb expects at least a full byte, it seems
        bit_size = 8;
        break;
    case TypeKind::Array: {
        auto* arr_type = static_cast<const ArrayType*>(ast_type);
        multi_int start{arr_type->index_type->range.lower_bound};
        multi_int end{arr_type->index_type->range.upper_bound};
        llvm::Metadata* subscript =
            m_dbg_builder.getOrCreateSubrange(to_int(std::move(start)),
                                              to_int(std::move(end)));
        llvm::DINodeArray array = m_dbg_builder.getOrCreateArray(subscript);
        // TODO: check if alignment of zero (arg #2) is always correct
        return m_dbg_builder.createArrayType(
            arr_type->index_type->range.size() * arr_type->bit_size(), 0,
            to_dbg_type(arr_type->element_type), array);
    }
    default:
        assert(ast_type == &Type::Void);
        return nullptr;
    }

    return m_dbg_builder.createBasicType(ast_type->name, bit_size, encoding);
}

llvm::DISubroutineType* DebugGenerator::to_dbg_type(const Function* ast_funct)
{
    llvm::SmallVector<llvm::Metadata*, 8> type_list;
    // Return type is always at index 0
    type_list.push_back(to_dbg_type(ast_funct->return_type));
    for(auto& arg : ast_funct->parameters) {
        type_list.push_back(to_dbg_type(arg->type));
    }

    return m_dbg_builder.createSubroutineType(m_dbg_builder.getOrCreateTypeArray(type_list));
}

void DebugGenerator::addFunction(llvm::IRBuilder<>& ir_builder,
                                 const Function* ast_funct, llvm::Function* funct)
{
    if(m_is_active) {
        llvm::DISubprogram* dbg_data = m_dbg_builder.createFunction(
            m_file, ast_funct->name, "", m_file, ast_funct->line_num(),
            to_dbg_type(ast_funct), ast_funct->line_num(), llvm::DINode::FlagZero,
            llvm::DISubprogram::SPFlagDefinition);
        funct->setSubprogram(dbg_data);
        m_scopes.push_back(dbg_data);
        // Tells debugger to skip over function prologue asm instructions
        ir_builder.SetCurrentDebugLocation(llvm::DebugLoc::get(0, 0, nullptr));
    }
}

void DebugGenerator::closeScope()
{
    if(m_is_active)
        m_scopes.pop_back();
}

void DebugGenerator::setLocation(unsigned int line_num, llvm::IRBuilder<>& ir_builder)
{
    if(m_is_active) {
        ir_builder.SetCurrentDebugLocation(
            llvm::DebugLoc::get(line_num, 1, m_scopes.back()));
    }
}

void DebugGenerator::addAutoVar(llvm::BasicBlock* block, llvm::Value* llvm_var,
                                const NamedLValue* var, unsigned int line_num)
{
    if(m_is_active) {
        llvm::DILocalVariable* dbg_var = m_dbg_builder.createAutoVariable(
            m_scopes.back(), var->name, m_file, line_num, to_dbg_type(var->type));
        m_dbg_builder.insertDeclare(
            llvm_var, dbg_var, m_dbg_builder.createExpression(),
            llvm::DebugLoc::get(line_num, 1, m_scopes.back()), block);
    }
}

void DebugGenerator::finalize()
{
    if(m_is_active)
        m_dbg_builder.finalize();
}

// Util functions
llvm::Value* truncate_to_bool(llvm::IRBuilder<>& ir_builder, llvm::Value* integer)
{
    return ir_builder.CreateTrunc(integer, llvm::Type::getInt1Ty(ir_builder.getContext()));
}

llvm::Type* CodeGenerator::to_llvm_type(const Type* ast_type)
{
    switch(ast_type->kind()) {
    case TypeKind::Range:
    case TypeKind::Boolean:
        return llvm::IntegerType::get(m_context, ast_type->bit_size());
    case TypeKind::Array: {
        auto* arr_type = static_cast<const ArrayType*>(ast_type);
        // TODO: when adding indexing support, add offset so element 0 lines
        // up with the array type's start index value
        return llvm::ArrayType::get(to_llvm_type(arr_type->element_type),
                                    arr_type->index_type->range.size());
    }
    case TypeKind::Ref: {
        auto* ref_type = static_cast<const RefType*>(ast_type);
        return llvm::PointerType::get(to_llvm_type(ref_type->inner_type), 0);
    }
    default:
        // TODO: add support for determining LLVM type for other AST types.
        // Note that literal types should never reach here (see the literal
        // codegen() functions)
        assert(false);
        return nullptr;
    }
}

llvm::ConstantInt* CodeGenerator::to_llvm_int(const multi_int& value,
                                              unsigned short bit_size)
{
    return llvm::ConstantInt::get(
        m_context, llvm::APInt(bit_size, value.str(), 10));
}

CodeGenerator::CodeGenerator(const char* source_filename,
                             std::vector<Magnum::Pointer<Function>>& functions,
                             std::vector<Magnum::Pointer<Initialization>>& global_vars,
                             Mode build_mode)
    : m_ir_builder(m_context), m_module(source_filename, m_context),
      m_functions(functions), m_global_vars(global_vars),
      m_dbg_gen(build_mode == Mode::Debug, m_module, source_filename),
      m_build_mode(build_mode)
{
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    const std::string target_triple{llvm::sys::getDefaultTargetTriple()};

    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple, error);
    if(!error.empty()) {
        Error().put("Codegen error:").quote(error).raise();
    }
    const llvm::TargetOptions options;

    m_target_machine = target->createTargetMachine(
        target_triple, "generic", "",
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
    return gen.m_ir_builder.getInt8(value);
}

llvm::Value* IntLiteral::codegen(CodeGenerator& gen)
{
    return gen.to_llvm_int(value, type()->bit_size());
}

llvm::Value* BoolLiteral::codegen(CodeGenerator& gen)
{
    return gen.m_ir_builder.getIntN(type()->bit_size(), value);
}

llvm::Value* FloatLiteral::codegen(CodeGenerator& gen)
{
    return llvm::ConstantFP::get(gen.m_context, llvm::APFloat(value));
}

llvm::Value* LValueExpression::codegen(CodeGenerator& gen)
{
    llvm::Value* src_lvalue = gen.m_lvalues[lvalue];
    assert(src_lvalue != nullptr);
    return gen.m_ir_builder.CreateLoad(src_lvalue, lvalue->name);
}

llvm::Value* RefExpression::codegen(CodeGenerator& gen)
{
    llvm::Value* value = gen.m_lvalues[lvalue];
    assert(value != nullptr);
    return value;
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
        return nullptr;
    }
}

llvm::Value* BinaryExpression::codegen(CodeGenerator& gen)
{
    bool type_is_signed = false;
    if(left->type()->kind() == TypeKind::Range) {
        type_is_signed = static_cast<const RangeType*>(left->type())->is_signed();
    } else if(right->type()->kind() == TypeKind::Range) {
        type_is_signed = static_cast<const RangeType*>(right->type())->is_signed();
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
        return nullptr;
    }
}

llvm::Value* FunctionCall::codegen(CodeGenerator& gen)
{
    llvm::Function* funct_to_call = gen.m_module.getFunction(name());
    assert(funct_to_call != nullptr);

    llvm::SmallVector<llvm::Value*, 6> args;
    for(auto& arg : arguments) {
        args.push_back(arg->codegen(gen));
    }
    llvm::CallInst* call_instr = gen.m_ir_builder.CreateCall(funct_to_call, args);
    // Can't name result of void function calls (they don't return anything)
    if(type() != &Type::Void)
        call_instr->setName("call" + name());
    return call_instr;
}

llvm::Value* IndexOp::codegen(CodeGenerator& gen)
{
    assert(base_expr->kind() == ExpressionKind::LValue);
    const NamedLValue* lval = static_cast<LValueExpression*>(base_expr.get())->lvalue;
    auto match = gen.m_lvalues.find(lval);
    assert(match == gen.m_lvalues.find(lval));
    llvm::Value* alloc = match->second;

    llvm::Value* offset_index = index_expr->codegen(gen);
    assert(offset_index->getType()->isIntegerTy());
    assert(index_expr->type()->kind() == TypeKind::Range);
    auto* index_type = static_cast<const RangeType*>(index_expr->type());
    // Since arrays can be indexed starting at any number, need to subtract
    // the starting value of this array's index type from the given index, since
    // in LLVM all arrays start at 0
    llvm::Value* gep_index =
        gen.m_ir_builder.CreateSub(offset_index,
                                   gen.to_llvm_int(index_type->range.lower_bound,
                                                   index_type->bit_size()));
    llvm::Value* indexes[] = { gen.m_ir_builder.getIntN(32, 0), gep_index };
    assert(indexes[1] != nullptr);
    assert(indexes[1]->getType()->isIntegerTy());
    auto* array_type = llvm::cast<llvm::ArrayType>(gen.to_llvm_type(lval->type));

    gen.m_dbg_gen.setLocation(line_num(), gen.m_ir_builder);
    auto* ptr = gen.m_ir_builder.CreateInBoundsGEP(array_type, alloc, indexes,
                                                   "arr_elem_ptr");

    return gen.m_ir_builder.CreateLoad(array_type->getElementType(), ptr, "arr_elem");
}

// Always returns nullptr; should only be called by store_expr_result()
llvm::Value* InitList::codegen(CodeGenerator& gen)
{
    assert(lvalue->kind() == LValueKind::Named);
    const auto match = gen.m_lvalues.find(static_cast<const NamedLValue*>(lvalue));
    assert(match != gen.m_lvalues.end());
    llvm::Value* alloc = match->second;
    gen.m_dbg_gen.setLocation(line, gen.m_ir_builder);

    llvm::Value* indexes[2] = { gen.m_ir_builder.getIntN(32, 0) };
    llvm::Type* array_type = gen.to_llvm_type(lvalue->type);
    for(uint64_t i = 0; i < values.size(); ++i) {
        // index 0: 0 bytes past the array ptr; index i: index into array
        indexes[1] = gen.m_ir_builder.getIntN(lvalue->type->bit_size(), i);
        // Get a pointer to to the current array element
        auto* element_ptr = gen.m_ir_builder.CreateInBoundsGEP(
            array_type, alloc, indexes, "elem_ptr");
        // Store the value at that array element
        gen.m_ir_builder.CreateStore(values[i]->codegen(gen), element_ptr);
    }
    return nullptr;
}

llvm::AllocaInst* CodeGenerator::prepend_alloca(llvm::Function* funct,
                                                llvm::Type* type, const std::string& name)
{
    auto prev_insert_point = m_ir_builder.saveIP();
    // All allocas need to be in the entry block
    // Allocas will be prepended to the entry block (so they all occur
    // before any are assigned values. This prevents LLVM passes from
    // creating new basic blocks and splitting up the allocas into different blocks)
    llvm::BasicBlock* entry = &funct->getEntryBlock();
    m_ir_builder.SetInsertPoint(entry, entry->begin());

    llvm::AllocaInst* alloc = m_ir_builder.CreateAlloca(type, 0, name.c_str());
    m_ir_builder.restoreIP(prev_insert_point);
    return alloc;
}

void CodeGenerator::store_expr_result(Expression* expr, llvm::Value* alloc)
{
    llvm::Value* input_instr = expr->codegen(*this);
    if(input_instr != nullptr) {
        // See InitList::codegen (which makes this store instruction redundant)
        m_ir_builder.CreateStore(input_instr, alloc);
    }
}

void CodeGenerator::add_lvalue_init(llvm::Function* function, Statement* statement)
{
    auto* init = static_cast<Initialization*>(statement);
    NamedLValue* lvalue = init->lvalue.get();

    llvm::AllocaInst* alloc = prepend_alloca(function, to_llvm_type(lvalue->type),
                                             lvalue->name);
    m_lvalues[lvalue] = alloc;

    if(init->expression != nullptr) {
        m_dbg_gen.setLocation(init->line_num(), m_ir_builder);
        m_dbg_gen.addAutoVar(m_ir_builder.GetInsertBlock(), alloc, lvalue, init->line_num());
        store_expr_result(init->expression.get(), alloc);
    }
}

void CodeGenerator::in_statement(llvm::Function* curr_funct, Statement* statement)
{
    switch(statement->kind()) {
    case StatementKind::Basic: {
        auto* curr_statement = static_cast<BasicStatement*>(statement);
        m_dbg_gen.setLocation(curr_statement->line_num(), m_ir_builder);
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
    case StatementKind::Return:
        in_return_statement(static_cast<ReturnStatement*>(statement));
        break;
    case StatementKind::Block:
        // TODO: add support for anonymous blocks
        break;
    }
}

void CodeGenerator::in_assignment(Assignment* assgn)
{
    LValue* lvalue = assgn->lvalue;
    llvm::Value* dest_ptr;
    switch(lvalue->kind()) {
    case LValueKind::Named: {
        auto match = m_lvalues.find(static_cast<const NamedLValue*>(lvalue));
        assert(match != m_lvalues.end());

        m_dbg_gen.setLocation(assgn->line_num(), m_ir_builder);
        dest_ptr = match->second;
        break;
    }
    case LValueKind::Index: {
        // Normally, accessing an array element means loading it, but in this
        // case, we don't want the loaded value; we want the pointer to the value
        auto* load_instr = llvm::cast<llvm::LoadInst>(
            static_cast<IndexLValue*>(lvalue)->array_access->codegen(*this));
        dest_ptr = load_instr->getPointerOperand();
        load_instr->eraseFromParent();
        break;
    }
    }

    store_expr_result(assgn->expression.get(), dest_ptr);
}

void CodeGenerator::in_return_statement(ReturnStatement* stmt)
{
    m_dbg_gen.setLocation(stmt->line_num(), m_ir_builder);
    if(stmt->expression.get() != nullptr) {
        m_ir_builder.CreateRet(stmt->expression->codegen(*this));
    } else {
        m_ir_builder.CreateRetVoid();
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
    bool is_if_block = false;
    if(successor == nullptr) {
        successor = llvm::BasicBlock::Create(m_context, "successor", curr_funct);
        is_if_block = true;
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
        m_dbg_gen.setLocation(ifblock->else_or_else_if->line_num(), m_ir_builder);
        in_if_block(curr_funct, static_cast<IfBlock*>(ifblock->else_or_else_if.get()),
                    successor);
    } else if(ifblock->else_or_else_if->kind() == StatementKind::Block) {
        // Else block
        if_false = llvm::BasicBlock::Create(m_context, "iffalse", curr_funct);
        m_ir_builder.SetInsertPoint(if_false);
        m_dbg_gen.setLocation(ifblock->else_or_else_if->line_num(), m_ir_builder);
        in_block(curr_funct, ifblock->else_or_else_if.get(), successor);
    }
    // Finally, insert the branch instruction right before the two branching blocks
    m_ir_builder.restoreIP(cond_br_point);
    m_ir_builder.CreateCondBr(condition, if_true, if_false);
    if(is_if_block && successor->hasNPredecessors(0)) {
        // Handle case in which all branches always return (meaning successor-block
        // is never reached). We have to check that we are currently processing
        // an if-block since the call to generate an if-block also handles creating
        // the successor block
        successor->eraseFromParent();
    } else {
        m_ir_builder.SetInsertPoint(successor);
    }
}

// Currently used to generate else-blocks, but should work for anonymous blocks
// if they are added in the future
void CodeGenerator::in_block(llvm::Function* curr_funct,
                             Block* block, llvm::BasicBlock* successor)
{
    for(auto& stmt : block->statements) {
        in_statement(curr_funct, stmt.get());
    }
    if(m_ir_builder.GetInsertBlock()->getTerminator() == nullptr) {
        // Only jump to successor if a ret wasn't already inserted
        m_ir_builder.CreateBr(successor);
    }
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

void CodeGenerator::declare_globals()
{
    for(auto& global : m_global_vars) {
        NamedLValue* lvalue = global->lvalue.get();
        llvm::Type* type = to_llvm_type(lvalue->type);
        llvm::Constant* init_val;
        if(global->expression == nullptr) {
            init_val = llvm::Constant::getNullValue(type);
        } else {
            init_val = llvm::cast<llvm::Constant>(global->expression->codegen(*this));
        }
        auto* global_ptr =
            new llvm::GlobalVariable(m_module, type, !lvalue->is_mutable,
                                     llvm::GlobalValue::LinkageTypes::InternalLinkage,
                                     init_val, lvalue->name);
        // Globals with same initializer will be merged
        // TODO: when pointers/exported symbols added, maybe change this?
        global_ptr->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
        m_lvalues[lvalue] = global_ptr;
    }
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
    llvm::SmallVector<llvm::Type*, 6> parameter_types;
    llvm::SmallVector<llvm::StringRef, 6> parameter_names;
    for(const auto& ast_function : m_functions) {
        auto linkage_type = llvm::Function::InternalLinkage;
        if(ast_function->kind() == FunctionKind::Builtin) {
            // TODO: maybe add debug info. for builtin functions?
            auto* builtin = static_cast<const BuiltinFunction*>(ast_function.get());
            if(!builtin->is_used)
                continue;
            linkage_type = llvm::Function::ExternalLinkage;
        } else if(ast_function->name == "main") {
            linkage_type = llvm::Function::ExternalLinkage;
        }

        for(const auto& ast_param : ast_function->parameters) {
            parameter_types.push_back(to_llvm_type(ast_param->type));
            parameter_names.push_back(ast_param->name);
        }

        auto* return_type = llvm::Type::getVoidTy(m_context);
        if(ast_function->return_type != &Type::Void) {
            return_type = to_llvm_type(ast_function->return_type);
        }
        auto* funct_type = llvm::FunctionType::get(return_type, parameter_types, false);
        auto* curr_funct = llvm::Function::Create(funct_type, linkage_type,
                                                  ast_function->name,
                                                  m_module);
        assert(curr_funct->getParent() != nullptr);
        // Disable generation of exception unwind tables (for now)
        curr_funct->addFnAttr(llvm::Attribute::NoUnwind);
        // Identical functions will be merged
        curr_funct->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

        auto param_name_it = parameter_names.begin();
        for(llvm::Argument& param : curr_funct->args()) {
            param.setName(*param_name_it);
            ++param_name_it;
        }

        parameter_types.clear();
        parameter_names.clear();
    }
}

void CodeGenerator::define_functions()
{
    for(auto& fcn : m_functions) {
        if(fcn->kind() != FunctionKind::Normal)
            // Builtin functions have no body
            continue;
        auto* function = static_cast<BBFunction*>(fcn.get());
        llvm::Function* curr_funct = m_module.getFunction(function->name);
        // Next, create a block containing the body of the function
        auto* funct_body = llvm::BasicBlock::Create(m_context, "entry", curr_funct);
        m_ir_builder.SetInsertPoint(funct_body);
        m_dbg_gen.addFunction(m_ir_builder, fcn.get(), curr_funct);

        auto ast_arg_it = fcn->parameters.begin();
        // When arguments are mutable, need to alloca them in the funct body in order
        // to read/write to them (since codegen assumes all variables are on stack)
        // TODO: For constant arguments, avoid the unnecessary alloca and have way
        //  where reads (e.g. LValueExpressions) do not try to load from arg ptr,
        //  instead using it directly, since it will already be in a register
        for(llvm::Argument& arg : curr_funct->args()) {
            llvm::AllocaInst* alloc = prepend_alloca(curr_funct, arg.getType(),
                                                     arg.getName());
            m_lvalues[ast_arg_it->get()] = alloc;
            ++ast_arg_it;
            m_ir_builder.CreateStore(&arg, alloc);
        }
        for(auto& statement : function->body.statements) {
            in_statement(curr_funct, statement.get());
        }
        if(fcn->return_type == &Type::Void
           && m_ir_builder.GetInsertBlock()->getTerminator() == nullptr) {
            // All blocks must end in a ret instruction of some kind
            m_ir_builder.CreateRetVoid();
        }
        m_dbg_gen.closeScope();
    }
}

void CodeGenerator::run()
{
    declare_function_headers();
    declare_globals();
    define_functions();

    m_dbg_gen.finalize();

    if(m_build_mode == Mode::Optimize) {
        optimize(m_module);
    }

    m_module.print(llvm::errs(), nullptr);
    assert(!llvm::verifyModule(m_module, &llvm::errs()));

    std::filesystem::path object_file{m_module.getSourceFileName()};
    object_file.replace_extension(".o");
    object_file = object_file.filename();
    emit(m_module, m_target_machine, object_file);
    link(object_file);
}
