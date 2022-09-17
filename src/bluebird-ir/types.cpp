#include "types.h"
#include "ast.h"

namespace bluebirdIR {

IntLiteralType IntLiteralType::get(mlir::MLIRContext* context)
{
    return Base::get(context);
}


RangeTypeStorage::RangeTypeStorage(const ::IntRangeType& ast_type) : ast_type(ast_type) {}

bool RangeTypeStorage::operator==(const KeyTy& key) const
{
    return &key == &ast_type;
}

llvm::hash_code RangeTypeStorage::hashKey(const KeyTy &key)
{
    return llvm::hash_value(key.name);
}

RangeTypeStorage* RangeTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
{
    return new (allocator.allocate<RangeTypeStorage>()) RangeTypeStorage(key);
}


RangeType RangeType::get(mlir::MLIRContext* context, const ::IntRangeType& ast_type)
{
    return Base::get(context, ast_type);
}

};
