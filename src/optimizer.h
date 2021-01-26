#ifndef OPTIMIZER_H
#define OPTIMIZER_H
namespace llvm {
    class Module;
};

void optimize(llvm::Module&);
#endif
