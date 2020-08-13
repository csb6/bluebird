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

#include "ast.h"
#include <unordered_map>
#include <vector>
#include <string>

class CodeGenerator {
private:
    llvm::LLVMContext m_context;
    llvm::IRBuilder<> m_ir_builder;
    Magnum::Pointer<llvm::Module> m_curr_module;

    const std::vector<Function>& m_functions;
    std::unordered_map<std::string, llvm::Type*> m_types;
public:
    CodeGenerator(const std::vector<Function>&, const std::vector<Type>&);
    void run();
};
#endif
