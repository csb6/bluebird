#ifndef BLUEBIRD_CODEGEN_H
#define BLUEBIRD_CODEGEN_H
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
#include <CorradePointer.h>

namespace Magnum = Corrade::Containers;

namespace llvm {
    class Value;
    class ConstantInt;
    class TargetMachine;
};

/* This file contains a class that can generate LLVM IR (and optionally debug
   info.) from a given AST. It expects the AST to be fully type-checked and
   for all types/function calls to be resolved to definitions. In other words,
   it expects the AST to have been run through the Checker beforehand.

   Ideally, the implementation of the file could be swapped out with a different
   code generator (such as GCC) without changing earlier stages of the compiler.
*/

/* Helper for building up debug information as the LLVM IR is
   being generated */
class DebugGenerator {
private:
    llvm::DIBuilder m_dbg_builder;
    llvm::DICompileUnit* m_dbg_unit;
    llvm::DIFile* m_file;
    // true if currently in debug mode
    bool m_is_active;
    std::vector<llvm::DIScope*> m_scopes;

    llvm::DIType* to_dbg_type(const struct Type* ast_type);
    llvm::DISubroutineType* to_dbg_type(const struct Function* ast_funct);
public:
    DebugGenerator(bool is_active, llvm::Module&, const char* source_filename);
    /* Register a new function in the debugger. Also opens a new scope. */
    void addFunction(llvm::IRBuilder<>&, const Function*, llvm::Function*);
    /* Mark the current scope as finished, pop it from the stack of scopes */
    void closeScope();
    /* Associate the current instruction with a source code line number */
    void setLocation(unsigned int line_num, llvm::IRBuilder<>&);
    /* Register a local (auto duration) variable in the debugger's current
       scope */
    void addAutoVar(llvm::BasicBlock*, llvm::Value*, const struct NamedLValue*,
                    unsigned int line_num);
    /* Resolve all entities in the debugger. Needs to be done before module
       is printed/generated */
    void finalize();
};

/* Generates, emits, and links into an executable an LLVM module using
   the AST as input */
class CodeGenerator {
public:
    // Choosen via command-line flags in main()
    enum class Mode {
        // No debug symbols, no optimizations
        Default,
        // Debug symbols, no optimizations
        Debug,
        // Optimizations, no debug symbols
        Optimize
    };
private:
    llvm::LLVMContext m_context;
    llvm::IRBuilder<> m_ir_builder;
    llvm::Module m_module;
    llvm::TargetMachine* m_target_machine;

    std::vector<Magnum::Pointer<Function>>& m_functions;
    std::vector<Magnum::Pointer<struct Initialization>>& m_global_vars;
    std::unordered_map<const NamedLValue*, llvm::Value*> m_lvalues;

    DebugGenerator m_dbg_gen;
    Mode m_build_mode;

    // For codegen, virtual functions attached to each Expression subclass.
    // These functions are defined in codegenerator.cpp
    friend struct StringLiteral;
    friend struct CharLiteral;
    friend struct IntLiteral;
    friend struct BoolLiteral;
    friend struct FloatLiteral;
    friend struct LValueExpression;
    friend struct RefExpression;
    friend struct UnaryExpression;
    friend struct BinaryExpression;
    friend struct FunctionCall;
    friend struct IndexOp;
    friend struct InitList;

    void declare_globals();
    void declare_builtin_functions();
    void declare_function_headers();
    void define_functions();
    llvm::Type* to_llvm_type(const Type* ast_type);
    llvm::ConstantInt* to_llvm_int(const class multi_int&, unsigned short bit_size);
    llvm::AllocaInst* prepend_alloca(llvm::Function*, llvm::Type*,
                                     const std::string& name);
    void store_expr_result(struct Expression*, llvm::Value* alloc);
    // Generate code for initializing lvalues
    void add_lvalue_init(llvm::Function*, struct Statement*);
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
public:
    /* Setup code generation for a given AST */
    CodeGenerator(const char* source_filename,
                  std::vector<Magnum::Pointer<Function>>&,
                  std::vector<Magnum::Pointer<Initialization>>& global_vars,
                  Mode build_mode = Mode::Default);
    /* Generate LLVM IR, optionally optimize it, emit the
       module into an object file, and then link the object file into
       an executable */
    void run();
};
#endif
