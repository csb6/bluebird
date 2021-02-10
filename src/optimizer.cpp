#include "optimizer.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/Sink.h>
#include <llvm/Transforms/Scalar/TailRecursionElimination.h>
#include <llvm/Transforms/Vectorize/SLPVectorizer.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/IPO/PartialInlining.h>
#include <llvm/Transforms/IPO/DeadArgumentElimination.h>
#include <llvm/Transforms/IPO/SCCP.h>
#include <llvm/Transforms/IPO/MergeFunctions.h>
#include <llvm/Transforms/IPO/GlobalOpt.h>
#pragma GCC diagnostic pop

void optimize(llvm::Module& module)
{
    llvm::FunctionPassManager funct_manager;
    // mem2reg
    funct_manager.addPass(llvm::PromotePass());
    // Aggressive Instruction Combine
    funct_manager.addPass(llvm::AggressiveInstCombinePass());
    // Reassociate (reorder expressions to improve constant propagation)
    funct_manager.addPass(llvm::ReassociatePass());
    // Eliminate common subexpressions
    funct_manager.addPass(llvm::GVN());
    // Loop Simplification (needs to be followed by a simplifycfg pass)
    funct_manager.addPass(llvm::LoopSimplifyPass());
    funct_manager.addPass(llvm::SimplifyCFGPass());
    // Aggressive Dead Code Elimination
    funct_manager.addPass(llvm::ADCEPass());
    // Sink (move instructions to successor blocks where possible)
    funct_manager.addPass(llvm::SinkingPass());
    // Tail Call Elimination
    funct_manager.addPass(llvm::TailCallElimPass());
    // Change series of stores into vector-stores
    funct_manager.addPass(llvm::SLPVectorizerPass());
    // Try to convert aggregates to multiple scalar allocas, then convert to SSA
    // where possible
    funct_manager.addPass(llvm::SROA());

    llvm::ModulePassManager module_manager;
    // Enables function passes to be used on a module
    module_manager.addPass(llvm::ModuleToFunctionPassAdaptor{std::move(funct_manager)});
    // Inlines parts of functions (e.g. if-statements if they surround a function body)
    module_manager.addPass(llvm::PartialInlinerPass());
    // Interprocedural constant propagation
    module_manager.addPass(llvm::IPSCCPPass());
    // Eliminates unused arguments and return values from functions
    module_manager.addPass(llvm::DeadArgumentEliminationPass());
    // Merge identical functions
    module_manager.addPass(llvm::MergeFunctionsPass());
    // Removes unused global variables
    module_manager.addPass(llvm::GlobalOptPass());

    llvm::FunctionAnalysisManager funct_analysis;
    llvm::ModuleAnalysisManager module_analysis;
    llvm::PassBuilder pass_builder;
    pass_builder.registerFunctionAnalyses(funct_analysis);
    pass_builder.registerModuleAnalyses(module_analysis);
    module_analysis.registerPass(
        [&] { return llvm::FunctionAnalysisManagerModuleProxy(funct_analysis); });
    funct_analysis.registerPass(
        [&] { return llvm::ModuleAnalysisManagerFunctionProxy(module_analysis); });

    module_manager.run(module, module_analysis);
}
