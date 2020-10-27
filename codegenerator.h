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
#ifndef BLUEBIRD_CODEGEN_H
#define BLUEBIRD_CODEGEN_H
// Internal LLVM code has lots of warnings that we don't
// care about, so ignore them
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#pragma GCC diagnostic pop

#include <vector>
#include <unordered_map>
#include <string>
#include <CorradePointer.h>

namespace Magnum = Corrade::Containers;

namespace llvm {
    class AllocaInst;
};

class CodeGenerator {
private:
    llvm::LLVMContext m_context;
    llvm::IRBuilder<> m_ir_builder;
    Magnum::Pointer<llvm::Module> m_curr_module;

    const std::vector<struct Function*>& m_functions;
    std::unordered_map<const struct LValue*, llvm::AllocaInst*> m_lvalues;

    // Generate code for expressions
    llvm::Value* in_expression(const struct Expression*);
    llvm::Value* in_string_literal(const Expression*);
    llvm::Value* in_char_literal(const Expression*);
    llvm::Value* in_int_literal(const Expression*);
    llvm::Value* in_float_literal(const Expression*);
    llvm::Value* in_lvalue_expression(const Expression*);
    llvm::Value* in_unary_expression(const Expression*);
    llvm::Value* in_binary_expression(const Expression*);
    llvm::Value* in_function_call(const Expression*);

    // Generate code for initializing lvalues
    void add_lvalue_init(llvm::Function*, const struct Statement*);
public:
    CodeGenerator(const std::vector<Function*>&);
    void run();
};
#endif
