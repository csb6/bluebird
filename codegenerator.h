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
