#ifndef BLUEBIRD_IR_DIALECT_H
#define BLUEBIRD_IR_DIALECT_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Attributes.h>
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


class BinaryOp {
public:
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    static void build(mlir::OpBuilder&, mlir::OperationState&, mlir::Value, mlir::Value);
};

class AddOp : public BinaryOp,
              public mlir::Op<AddOp,
                              mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::IsCommutative,
                              mlir::OpTrait::OneResult,
                              mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;

    static llvm::StringRef getOperationName() { return "bluebirdIR.add"; }
};

class SubtractOp : public BinaryOp,
                   public mlir::Op<SubtractOp,
                                   mlir::OpTrait::NOperands<2>::Impl,
                                   mlir::OpTrait::OneResult,
                                   mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;

    static llvm::StringRef getOperationName() { return "bluebirdIR.sub"; }
};

class MultiplyOp : public BinaryOp,
                   public mlir::Op<MultiplyOp,
                                   mlir::OpTrait::NOperands<2>::Impl,
                                   mlir::OpTrait::IsCommutative,
                                   mlir::OpTrait::OneResult,
                                   mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;

    static llvm::StringRef getOperationName() { return "bluebirdIR.mult"; }
};

class DivideOp : public BinaryOp,
                 public mlir::Op<DivideOp,
                                 mlir::OpTrait::NOperands<2>::Impl,
                                 mlir::OpTrait::OneResult,
                                 mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;

    static llvm::StringRef getOperationName() { return "bluebirdIR.div"; }
};

class UnaryOp {
public:
    static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

    static void build(mlir::OpBuilder&, mlir::OperationState&, mlir::Value);
};

class NegateOp : public UnaryOp,
                 public mlir::Op<NegateOp,
                                 mlir::OpTrait::OneOperand,
                                 mlir::OpTrait::OneResult,
                                 mlir::OpTrait::SameOperandsAndResultType> {
public:
    using Op::Op;

    static llvm::StringRef getOperationName() { return "bluebirdIR.neg"; }
};

}
#endif
