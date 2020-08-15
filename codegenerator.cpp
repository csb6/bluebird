#include "codegenerator.h"
// Internal LLVM code has lots of warnings that we don't
// care about, so ignore them
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Verifier.h>
#pragma GCC diagnostic pop
#include <type_traits>

CodeGenerator::CodeGenerator(const std::vector<Function>& functions,
                             const std::vector<Type>& types)
    : m_ir_builder(m_context), m_functions(functions)
{
    // Translate the specific type names from the parser (owner of the types variable)
    // to the machine representation, since that is what LLVM needs. All typechecking
    // will be done before this stage, so assume all types are correct.
    for(const Type& type : types) {
        // TODO: take into account properties of each type in order to make it
        // have the right implementation (e.g. an array, character, etc.)
        //   Right now, assume all types are 32-bit integer types
        auto* number_type = llvm::IntegerType::get(m_context, 32);
        m_types[type.name] = number_type;
    }
}

llvm::Value* CodeGenerator::in_expression(const Expression* expression)
{
    switch(expression->type()) {
    case ExpressionType::StringLiteral:
        return in_string_literal(expression);
    case ExpressionType::CharLiteral:
        return in_char_literal(expression);
    case ExpressionType::IntLiteral:
        return in_int_literal(expression);
    case ExpressionType::FloatLiteral:
        return in_float_literal(expression);
    case ExpressionType::LValue:
        return in_lvalue_expression(expression);
    case ExpressionType::Binary:
        return in_binary_expression(expression);
    case ExpressionType::FunctionCall:
        return in_function_call(expression);
    }
}

llvm::Value* CodeGenerator::in_string_literal(const Expression* expression)
{
    auto* literal = static_cast<const Literal<StringLiteral_t>*>(expression);
    // TODO: make sure this doesn't mess-up the insertion point for IR instructions
    return m_ir_builder.CreateGlobalStringPtr(literal->value);
}

llvm::Value* CodeGenerator::in_char_literal(const Expression* expression)
{
    auto* literal = static_cast<const Literal<CharLiteral_t>*>(expression);
    const CharLiteral_t value = literal->value;
    // Create a signed `char` type
    return llvm::ConstantInt::get(m_context,
                                  llvm::APInt(sizeof(value), value, true));
}

llvm::Value* CodeGenerator::in_int_literal(const Expression* expression)
{
    auto* literal = static_cast<const Literal<IntLiteral_t>*>(expression);
    const IntLiteral_t value = literal->value;
    // TODO: determine signed-ness; right now, assume signed
    return llvm::ConstantInt::get(m_context,
                                  llvm::APInt(sizeof(value), value, true));
}

llvm::Value* CodeGenerator::in_float_literal(const Expression* expression)
{
    auto* literal = static_cast<const Literal<FloatLiteral_t>*>(expression);
    return llvm::ConstantFP::get(m_context, llvm::APFloat(literal->value));
}

llvm::Value* CodeGenerator::in_lvalue_expression(const Expression*)
{
    return nullptr;
}

llvm::Value* CodeGenerator::in_binary_expression(const Expression*)
{
    return nullptr;
}

llvm::Value* CodeGenerator::in_function_call(const Expression*)
{
    return nullptr;
}

void CodeGenerator::run()
{
    {
        // Reused between iterations to reduce allocations
        std::vector<llvm::Type*> parameter_types;
        std::vector<llvm::StringRef> parameter_names;
        for(const Function& function : m_functions) {
            // First, create a function declaration
            parameter_types.reserve(function.parameters.size());
            parameter_names.reserve(function.parameters.size());

            for(const auto& param : function.parameters) {
                parameter_types.push_back(m_types[param->type]);
                parameter_names.push_back(param->name);
            }

            // TODO: add support for return types other than void
            auto* funct_type = llvm::FunctionType::get(llvm::Type::getVoidTy(m_context),
                                                       parameter_types, false);
            // TODO: add more fine-grained support for Linkage
            auto* curr_funct = llvm::Function::Create(funct_type,
                                                      llvm::Function::ExternalLinkage,
                                                      function.name,
                                                      m_curr_module.get());

            auto param_name_it = parameter_names.begin();
            for(auto& param : curr_funct->args()) {
                param.setName(*param_name_it);
                ++param_name_it;
            }

            parameter_types.clear();
            parameter_names.clear();

            // Next, create a block containing the body of the function
            auto* funct_body = llvm::BasicBlock::Create(m_context, "body", curr_funct);
            m_ir_builder.SetInsertPoint(funct_body);
            for(const auto& statement : function.statements) {
                switch(statement->type()) {
                case StatementType::Basic: {
                    auto* curr_statement = static_cast<const BasicStatement*>(statement.get());
                    //in_expression(curr_statement->expression.get());
                    break;
                }
                default:
                    // TODO: add support for if-blocks, variable initializations, etc.
                    break;
                }
            }
        }
    }
} 
