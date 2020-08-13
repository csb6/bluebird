#include "codegenerator.h"
// Internal LLVM code has lots of warnings that we don't
// care about, so ignore them
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Verifier.h>
#pragma GCC diagnostic pop

CodeGenerator::CodeGenerator(const std::vector<Function>& functions,
                             const std::vector<Type>& types)
    : m_ir_builder(m_context), m_functions(functions)
{
    // Translate the specific type names from the parser (owner of the types variable)
    // to the machine representation, since that is what LLVM needs. All typechecking
    // will be done before this stage, so assume all types are correct.
    for(const Type& type : types) {
        // TODO: take into account properties of each type in order to make it
        // have the right implementation (e.g. an array, character, etc.)
        //   Right now, assume all types are 32-bit integer types
        auto* number_type = llvm::IntegerType::get(m_context, 32);
        m_types[type.name] = number_type;
    }
}

void CodeGenerator::run()
{
    {
        // Reused between iterations to reduce allocations
        std::vector<llvm::Type*> parameter_types;
        std::vector<llvm::StringRef> parameter_names;
        for(const Function& function : m_functions) {
            parameter_types.reserve(function.parameters.size());
            parameter_names.reserve(parameter_types.size());

            for(const auto& param : function.parameters) {
                parameter_types.push_back(m_types[param->type]);
                parameter_names.push_back(param->name);
            }

            // TODO: add support for return types other than void
            auto* funct_type = llvm::FunctionType::get(llvm::Type::getVoidTy(m_context),
                                                       parameter_types, false);
            // TODO: add more fine-grained support for Linkage
            auto* curr_funct = llvm::Function::Create(funct_type,
                                                      llvm::Function::ExternalLinkage,
                                                      function.name,
                                                      m_curr_module.get());

            auto param_name_it = parameter_names.begin();
            for(auto& param : curr_funct->args()) {
                param.setName(*param_name_it);
                ++param_name_it;
            }

            parameter_types.clear();
            parameter_names.clear();
        }
    }
} 
