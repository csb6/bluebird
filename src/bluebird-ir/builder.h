#ifndef BLUEBIRD_IR_BUILDER_H
#define BLUEBIRD_IR_BUILDER_H
#include <unordered_map>
#include "context.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#pragma GCC diagnostic pop

struct Assignable;
struct Variable;
struct Function;
struct Type;
struct Initialization;
struct IndexedVariable;

namespace bluebirdIR {

class Builder {
public:
    Builder(std::vector<Magnum::Pointer<Function>>&,
            std::vector<Magnum::Pointer<Type>>&,
            std::vector<Magnum::Pointer<Initialization>>& global_vars,
            std::vector<Magnum::Pointer<IndexedVariable>>&);
    void run();
private:
    mlir::MLIRContext m_context;
    mlir::ModuleOp m_module;
    mlir::OpBuilder m_builder;

    std::unordered_map<const Assignable*, mlir::Value> m_sse_vars;
    std::unordered_map<const Function*, mlir::FuncOp> m_mlir_functions;
    std::vector<Magnum::Pointer<Function>>& m_functions;
    std::vector<Magnum::Pointer<Type>>& m_types;
    std::vector<Magnum::Pointer<Initialization>>& m_global_vars;
    std::vector<Magnum::Pointer<IndexedVariable>>& m_index_vars;
};

};
#endif
