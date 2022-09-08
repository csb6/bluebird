#ifndef BLUEBIRD_IR_DIALECT_H
#define BLUEBIRD_IR_DIALECT_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#pragma GCC diagnostic pop

class multi_int;

namespace bluebirdIR {

class BluebirdIRDialect : public mlir::Dialect {
public:
    explicit BluebirdIRDialect(mlir::MLIRContext*);

    static llvm::StringRef getDialectNamespace() { return "bluebirdIR"; }
    void printType(mlir::Type, mlir::DialectAsmPrinter&) const override;
    void printAttribute(mlir::Attribute, mlir::DialectAsmPrinter&) const override;
};


struct IntLiteralAttrStorage;

class IntLiteralAttr : public mlir::Attribute::AttrBase<IntLiteralAttr,
                                                        mlir::Attribute,
                                                        IntLiteralAttrStorage> {
public:
    using Base::Base;
    static IntLiteralAttr get(mlir::MLIRContext*, const multi_int& value);

    const multi_int& getValue() const;
};


template<typename ConcreteOp, template <typename T> class... Traits>
class BluebirdOp : public mlir::Op<ConcreteOp,
                                   Traits...,
                                   mlir::OpTrait::OneResult,
                                   mlir::MemoryEffectOpInterface::Trait>
{
public:
    using mlir::Op<ConcreteOp,
        Traits...,
        mlir::OpTrait::OneResult,
        mlir::MemoryEffectOpInterface::Trait>::Op;

    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    // No side effects
    void getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>&) const {}
};

template<typename ConcreteOp, template <typename T> class... Traits>
class ConstantOp : public BluebirdOp<ConcreteOp,
                                     Traits...,
                                     mlir::OpTrait::ZeroOperands,
                                     mlir::OpTrait::ConstantLike> {
public:
    using BluebirdOp<ConcreteOp, Traits..., mlir::OpTrait::ZeroOperands,
                     mlir::OpTrait::ConstantLike>::BluebirdOp;
};

class BoolConstantOp : public ConstantOp<BoolConstantOp> {
public:
    using ConstantOp::ConstantOp;

    static llvm::StringRef getOperationName() { return "bluebirdIR.boolConstant"; }

    static void build(mlir::OpBuilder&, mlir::OperationState&, bool value, mlir::Type);
};

class CharConstantOp : public ConstantOp<CharConstantOp> {
public:
    using ConstantOp::ConstantOp;

    static llvm::StringRef getOperationName() { return "bluebirdIR.charConstant"; }

    static void build(mlir::OpBuilder&, mlir::OperationState&, char value, mlir::Type);
};

class IntConstantOp : public ConstantOp<IntConstantOp> {
public:
    using ConstantOp::ConstantOp;

    static llvm::StringRef getOperationName() { return "bluebirdIR.intConstant"; }

    static void build(mlir::OpBuilder&, mlir::OperationState&,
                      const multi_int& value, mlir::Type);
};

void build_binary_op(mlir::OperationState&, mlir::Value, mlir::Value);

template<typename ConcreteOp, template <typename T> class... Traits>
class BinaryOp : public BluebirdOp<ConcreteOp,
                                   Traits...,
                                   mlir::OpTrait::NOperands<2>::Impl> {
public:
    using BluebirdOp<ConcreteOp, Traits..., mlir::OpTrait::NOperands<2>::Impl>::BluebirdOp;

    static void build(mlir::OpBuilder&, mlir::OperationState& state,
                      mlir::Value operand1, mlir::Value operand2)
    {
        build_binary_op(state, operand1, operand2);
    }
};

class DivideOp : public BinaryOp<DivideOp,
                                 mlir::OpTrait::SameOperandsAndResultType> {
public:
    using BinaryOp::BinaryOp;

    static llvm::StringRef getOperationName() { return "bluebirdIR.div"; }
};


void build_unary_op(mlir::OperationState&, mlir::Value);

template<typename ConcreteOp, template <typename T> class... Traits>
class UnaryOp : public BluebirdOp<ConcreteOp,
                                  Traits...,
                                  mlir::OpTrait::OneOperand> {
public:
    using BluebirdOp<ConcreteOp, Traits..., mlir::OpTrait::OneOperand>::BluebirdOp;

    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    static void build(mlir::OpBuilder&, mlir::OperationState& state, mlir::Value operand)
    {
        build_unary_op(state, operand);
    }
};

class NegateOp : public UnaryOp<NegateOp,
                                mlir::OpTrait::IsIdempotent,
                                mlir::OpTrait::SameOperandsAndResultType> {
public:
    using UnaryOp::UnaryOp;

    static llvm::StringRef getOperationName() { return "bluebirdIR.neg"; }
};

class NotOp : public UnaryOp<NotOp,
                             mlir::OpTrait::IsIdempotent,
                             mlir::OpTrait::SameOperandsAndResultType,
                             mlir::OpTrait::ResultsAreBoolLike> {
public:
    using UnaryOp::UnaryOp;

    static llvm::StringRef getOperationName() { return "bluebirdIR.not"; }
};

}
#endif
