#include "dialect.h"
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
    addOperations<BoolConstantOp, CharConstantOp, IntConstantOp, FloatConstantOp,
                  NegateOp, NotOp>();
    addTypes<RangeType, IntLiteralType>();
    addAttributes<IntLiteralAttr>();
}

void BluebirdIRDialect::printType(mlir::Type, mlir::DialectAsmPrinter&) const
{}

void BluebirdIRDialect::printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer) const
{
    if(auto intLiteral = attr.dyn_cast<IntLiteralAttr>()) {
        printer << "intLiteral = " << intLiteral.getValue().str();
    } else {
        attr.print(printer.getStream());
    }
}


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


void BoolConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, bool value)
{
    auto attr = builder.getBoolAttr(value);
    state.addAttribute("value", attr);
    state.types.push_back(attr.getType());
}

void CharConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, char value)
{
    state.addAttribute("value", builder.getI8IntegerAttr(value));
    state.types.push_back(builder.getI8Type());
}

void IntConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                          const multi_int& value, mlir::Type type)
{
    state.addAttribute("value", IntLiteralAttr::get(builder.getContext(), value));
    state.types.push_back(type);
}

void FloatConstantOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, float value)
{
    state.addAttribute("value", builder.getF32FloatAttr(value));
    state.types.push_back(builder.getF32Type());
}


void build_binary_op(mlir::OperationState& state,
                     mlir::Value operand1, mlir::Value operand2)
{
    state.types.push_back(operand1.getType());
    state.operands.push_back(std::move(operand1));
    state.operands.push_back(std::move(operand2));
}

void build_unary_op(mlir::OperationState& state, mlir::Value operand)
{
    state.types.push_back(operand.getType());
    state.operands.push_back(std::move(operand));
}

};
