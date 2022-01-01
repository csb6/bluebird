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


struct CharLiteralAttrStorage;

class CharLiteralAttr : public mlir::Attribute::AttrBase<CharLiteralAttr,
                                                         mlir::Attribute,
                                                         CharLiteralAttrStorage> {
public:
    using Base::Base;
    static CharLiteralAttr get(mlir::MLIRContext*, char value);

    char getValue() const;
};


template<typename ConcreteOp, template <typename T> class... Traits>
class BluebirdOp : public mlir::Op<ConcreteOp,
                                   mlir::OpTrait::OneResult,
                                   mlir::MemoryEffectOpInterface::Trait,
                                   Traits...>
{
public:
    using mlir::Op<ConcreteOp,
        mlir::OpTrait::OneResult,
        mlir::MemoryEffectOpInterface::Trait,
        Traits...>::Op;

    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    void getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>&) const {}
};

void build_binary_op(mlir::OperationState&, mlir::Value, mlir::Value);

template<typename ConcreteOp, template <typename T> class... Traits>
class BinaryOp : public BluebirdOp<ConcreteOp,
                                   mlir::OpTrait::NOperands<2>::Impl,
                                   Traits...> {
public:
    using BluebirdOp<ConcreteOp, mlir::OpTrait::NOperands<2>::Impl, Traits...>::BluebirdOp;

    static void build(mlir::OpBuilder&, mlir::OperationState& state,
                      mlir::Value operand1, mlir::Value operand2)
    {
        build_binary_op(state, operand1, operand2);
    }
};

class AddOp : public BinaryOp<AddOp,
                              mlir::OpTrait::IsCommutative,
                              mlir::OpTrait::SameOperandsAndResultType> {
public:
    using BinaryOp::BinaryOp;

    static llvm::StringRef getOperationName() { return "bluebirdIR.add"; }
};

class SubtractOp : public BinaryOp<SubtractOp,
                                   mlir::OpTrait::SameOperandsAndResultType> {
public:
    using BinaryOp::BinaryOp;

    static llvm::StringRef getOperationName() { return "bluebirdIR.sub"; }
};

class MultiplyOp : public BinaryOp<MultiplyOp,
                                   mlir::OpTrait::IsCommutative,
                                   mlir::OpTrait::SameOperandsAndResultType> {
public:
    using BinaryOp::BinaryOp;

    static llvm::StringRef getOperationName() { return "bluebirdIR.mult"; }
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
                                  mlir::OpTrait::OneOperand,
                                  Traits...> {
public:
    using BluebirdOp<ConcreteOp, mlir::OpTrait::OneOperand, Traits...>::BluebirdOp;

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

}
#endif
