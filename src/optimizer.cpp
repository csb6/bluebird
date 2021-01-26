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
#pragma GCC diagnostic pop

void optimize(llvm::Module& module)
{
    llvm::FunctionPassManager funct_opt;
    // mem2reg
    funct_opt.addPass(llvm::PromotePass());
    // Aggressive Dead Code Elimination
    funct_opt.addPass(llvm::ADCEPass());
    // Aggressive Instruction Combine
    funct_opt.addPass(llvm::AggressiveInstCombinePass());
    // Reassociate (reorder expressions to improve constant propagation)
    funct_opt.addPass(llvm::ReassociatePass());
    // Eliminate common subexpressions
    funct_opt.addPass(llvm::GVN());
    // Loop Simplification (needs to be followed by a simplifycfg pass)
    funct_opt.addPass(llvm::LoopSimplifyPass());
    funct_opt.addPass(llvm::SimplifyCFGPass());
    // Sink (move instructions to successor blocks where possible)
    funct_opt.addPass(llvm::SinkingPass());
    // Tail Call Elimination
    funct_opt.addPass(llvm::TailCallElimPass());

    llvm::FunctionAnalysisManager funct_mng;
    llvm::PassBuilder pass_builder;
    pass_builder.registerFunctionAnalyses(funct_mng);
    for(llvm::Function& funct : module) {
        funct_opt.run(funct, funct_mng);
    }
}
