#include "dialect.h"
#include "types.h"
#include "multiprecision.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#pragma GCC diagnostic pop

namespace bluebirdIR {

BluebirdIRDialect::BluebirdIRDialect(mlir::MLIRContext* context)
    : mlir::Dialect(getDialectNamespace(), context, mlir::TypeID::get<BluebirdIRDialect>())
{
    addOperations<AddOp, SubtractOp, MultiplyOp, DivideOp, NegateOp>();
    addTypes<RangeType, IntLiteralType>();
    addAttributes<IntLiteralAttr, CharLiteralAttr>();
}

void BluebirdIRDialect::printType(mlir::Type, mlir::DialectAsmPrinter&) const
{}

void BluebirdIRDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer) const
{
    if(auto intLiteral = attr.dyn_cast<IntLiteralAttr>()) {
        printer << "bluebirdIR.intLiteral = " << intLiteral.getValue().str();
    } else if(auto charLiteral = attr.dyn_cast<CharLiteralAttr>()) {
        printer << "bluebirdIR.charLiteral = '" << charLiteral.getValue() << "'";
    } else {
        attr.print(printer.getStream());
    }
}


struct CharLiteralAttrStorage : public mlir::AttributeStorage {
    using KeyTy = char;

    explicit CharLiteralAttrStorage(char value) : value(value) {}

    bool operator==(const KeyTy& key) const
    {
        return key == value;
    }

    static CharLiteralAttrStorage* construct(mlir::AttributeStorageAllocator& allocator, const KeyTy& key)
    {
        return new (allocator.allocate<IntLiteralAttrStorage>()) CharLiteralAttrStorage(key);
    }

    char value;
};

CharLiteralAttr CharLiteralAttr::get(mlir::MLIRContext* context, char value)
{
    return Base::get(context, value);
}

char CharLiteralAttr::getValue() const { return getImpl()->value; }


struct IntLiteralAttrStorage : public mlir::AttributeStorage {
    using KeyTy = multi_int;

    explicit IntLiteralAttrStorage(const multi_int& value) : value(value) {}

    bool operator==(const KeyTy& key) const
    {
        return key == value;
    }

    static llvm::hash_code hashKey(const KeyTy& key)
    {
        return llvm::hash_value(key.str());
    }

    static IntLiteralAttrStorage* construct(mlir::AttributeStorageAllocator& allocator, const KeyTy& key)
    {
        return new (allocator.allocate<IntLiteralAttrStorage>()) IntLiteralAttrStorage(key);
    }

    multi_int value;
};

IntLiteralAttr IntLiteralAttr::get(mlir::MLIRContext* context, const multi_int& value)
{
    return Base::get(context, value);
}

const multi_int& IntLiteralAttr::getValue() const
{
    return getImpl()->value;
}


void BinaryOp::build(mlir::OpBuilder&, mlir::OperationState& state,
                     mlir::Value operand1, mlir::Value operand2)
{
    state.operands.push_back(operand1);
    state.operands.push_back(operand2);
    state.types.push_back(operand1.getType());
}


void UnaryOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::Value operand)
{
    state.operands.push_back(std::move(operand));
}

};
