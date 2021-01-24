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
#ifndef BLUEBIRD_CODEGEN_H
#define BLUEBIRD_CODEGEN_H
// Internal LLVM code has lots of warnings that we don't care about
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DIBuilder.h>
#pragma GCC diagnostic pop

#include <vector>
#include <unordered_map>
#include <filesystem>
#include <CorradePointer.h>

namespace Magnum = Corrade::Containers;

namespace llvm {
    class Module;
    class Value;
    class raw_fd_ostream;
    class TargetMachine;
    class DICompileUnit;
    class DIFile;
    class DIType;
    class DISubroutineType;
    class DIScope;
};

class DebugGenerator {
private:
    llvm::DIBuilder m_dbg_builder;
    llvm::DICompileUnit* m_dbg_unit;
    llvm::DIFile* m_file;
    bool m_is_active;
    std::vector<llvm::DIScope*> m_scopes;

    llvm::DIType* to_dbg_type(const struct Type* ast_type);
    llvm::DISubroutineType* to_dbg_type(const struct Function* ast_funct);
public:
    DebugGenerator(bool is_active, llvm::Module&, const char* source_filename);
    void addFunction(const Function*, llvm::Function*);
    void closeFunction(llvm::Function*);
    void closeScope();
    void setLocation(const struct Statement*, llvm::IRBuilder<>&);
    void finalize();
};

class CodeGenerator {
private:
    llvm::LLVMContext m_context;
    llvm::IRBuilder<> m_ir_builder;
    llvm::Module m_module;
    llvm::TargetMachine* m_target_machine;

    std::vector<Magnum::Pointer<Function>>& m_functions;
    std::vector<Magnum::Pointer<struct Initialization>>& m_global_vars;
    std::unordered_map<const struct LValue*, llvm::Value*> m_lvalues;

    DebugGenerator m_dbg_gen;

    // For codegen, virtual functions attached to each Expression subclass.
    // These functions are defined in codegenerator.cpp
    friend struct StringLiteral;
    friend struct CharLiteral;
    friend struct IntLiteral;
    friend struct FloatLiteral;
    friend struct LValueExpression;
    friend struct UnaryExpression;
    friend struct BinaryExpression;
    friend struct FunctionCall;

    void declare_globals();
    void declare_builtin_functions();
    void declare_function_headers();
    void define_functions();
    llvm::Type* to_llvm_type(const Type* ast_type);
    llvm::AllocaInst* prepend_alloca(llvm::Function*, llvm::Type*,
                                     const std::string& name);
    // Generate code for initializing lvalues
    void add_lvalue_init(llvm::Function*, Statement*);
    void in_statement(llvm::Function*, Statement*);
    void in_assignment(struct Assignment*);
    void in_return_statement(struct ReturnStatement*);
    // successor is nullptr -> if-block
    // sucessor isn't nullptr -> else-if-block
    void in_if_block(llvm::Function*, struct IfBlock*,
                     llvm::BasicBlock* successor = nullptr);
    void in_block(llvm::Function*, struct Block*,
                  llvm::BasicBlock* successor);
    void in_while_loop(llvm::Function*, struct WhileLoop*);

    void emit(const std::filesystem::path& object_file);
    void link(const std::filesystem::path& object_file,
              const std::filesystem::path& exe_file = "a.out");
public:
    CodeGenerator(const char* source_filename,
                  std::vector<Magnum::Pointer<Function>>&,
                  std::vector<Magnum::Pointer<Initialization>>&,
                  bool debug_mode = false);
    void run();
};
#endif
