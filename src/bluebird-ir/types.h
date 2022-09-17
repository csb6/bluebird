#ifndef BLUEBIRD_IR_TYPES_H
#define BLUEBIRD_IR_TYPES_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <mlir/IR/Types.h>
#pragma GCC diagnostic pop

// From the AST
struct IntRangeType;

namespace bluebirdIR {

class IntLiteralType : public mlir::Type::TypeBase<IntLiteralType, mlir::Type, mlir::TypeStorage> {
public:
    using Base::Base;

    static IntLiteralType get(mlir::MLIRContext*);
};

struct RangeTypeStorage : public mlir::TypeStorage {
    using KeyTy = ::IntRangeType;

    explicit RangeTypeStorage(const ::IntRangeType& ast_type);

    bool operator==(const KeyTy&) const;
    static llvm::hash_code hashKey(const KeyTy &);
    static RangeTypeStorage* construct(mlir::TypeStorageAllocator&, const KeyTy&);

    const ::IntRangeType& ast_type;
};

class RangeType : public mlir::Type::TypeBase<RangeType, mlir::Type, RangeTypeStorage> {
public:
    using Base::Base;

    static RangeType get(mlir::MLIRContext*, const ::IntRangeType& ast_type);
};

};
#endif
