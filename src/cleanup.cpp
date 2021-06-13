#include "cleanup.h"
#include "ast.h"
#include "error.h"

Cleanup::Cleanup(std::vector<Magnum::Pointer<Function>>& functions,
                 std::vector<Magnum::Pointer<Initialization>>& global_vars)
    : m_functions(functions), m_global_vars(global_vars)
{}

static
void fold_unary_constants(Magnum::Pointer<Expression>& expr_location_out,
                          UnaryExpression* unary_expr)
{
    auto replace_with_literal =
        [&](Expression* literal) {
            auto* released = unary_expr->right.release();
            assert(released == literal);
            expr_location_out.reset(literal);
        };

    switch(unary_expr->right->kind()) {
    case ExprKind::IntLiteral: {
        auto* r_int = static_cast<IntLiteral*>(unary_expr->right.get());
        switch(unary_expr->op) {
        case TokenType::Op_Minus:
            // TODO: use r_int->value.ones_complement(); for certain unsigned integer types
            r_int->value.negate();
            break;
        default:
            raise_error_expected("unary expression with operator that works on "
                                 "integer literals", unary_expr);
        }
        replace_with_literal(r_int);
        break;
    }
    case ExprKind::BoolLiteral: {
        auto* r_bool = static_cast<BoolLiteral*>(unary_expr->right.get());
        if(unary_expr->op == TokenType::Op_Not) {
            r_bool->value = !r_bool->value;
        } else {
            raise_error_expected("unary expression with logical operator "
                                 "(e.g. `not`)", unary_expr);
        }
        replace_with_literal(r_bool);
        break;
    }
    default:
        // No constant folding needed
        break;
    }
}

static
void fold_binary_constants(Magnum::Pointer<Expression>& expr_location_out,
                           BinaryExpression* bin_expr)
{
    auto replace_with_literal =
        [&](Expression* literal) {
            auto* released = bin_expr->left.release();
            assert(released == literal);
            expr_location_out.reset(literal);
        };

    const auto left_kind = bin_expr->left->kind();
    const auto right_kind = bin_expr->right->kind();
    if(left_kind == ExprKind::IntLiteral && right_kind == ExprKind::IntLiteral) {
        // Fold into a single literal (uses arbitrary-precision arithmetic)
        auto* l_int = static_cast<IntLiteral*>(bin_expr->left.get());
        auto* r_int = static_cast<IntLiteral*>(bin_expr->right.get());
        switch(bin_expr->op) {
        case TokenType::Op_Plus:
            l_int->value += r_int->value;
            break;
        case TokenType::Op_Minus:
            l_int->value -= r_int->value;
            break;
        case TokenType::Op_Mult:
            l_int->value *= r_int->value;
            break;
        case TokenType::Op_Div:
            l_int->value /= r_int->value;
            break;
        case TokenType::Op_Mod:
            l_int->value.mod(r_int->value);
            break;
        case TokenType::Op_Rem:
            l_int->value.rem(r_int->value);
            break;
        // TODO: Add support for shift operators
        case TokenType::Op_Thru:
        case TokenType::Op_Upto:
            // Can't do folds here, need to preserve left/right sides for a range
            return;
        default:
            if(is_bool_op(bin_expr->op)) {
                raise_error_expected("binary expression with an operator that works "
                                     "on integer literals", bin_expr);
            }
            return;
        }
        replace_with_literal(l_int);
    } else if(left_kind == ExprKind::BoolLiteral && right_kind == ExprKind::BoolLiteral) {
        // Fold into a single literal
        auto* l_bool = static_cast<BoolLiteral*>(bin_expr->left.get());
        auto* r_bool = static_cast<BoolLiteral*>(bin_expr->right.get());
        switch(bin_expr->op) {
        case TokenType::Op_And:
            l_bool->value &= r_bool->value;
            break;
        case TokenType::Op_Or:
            l_bool->value |= r_bool->value;
            break;
        case TokenType::Op_Xor:
            l_bool->value ^= r_bool->value;
            break;
        default:
            raise_error_expected("binary expression with logical operator", bin_expr);
        }
        replace_with_literal(l_bool);
    }
}

static
void fold_constants(Magnum::Pointer<Expression>& child_location)
{
    child_location->cleanup();
    if(child_location->kind() == ExprKind::Binary) {
        fold_binary_constants(child_location,
                              static_cast<BinaryExpression*>(child_location.get()));
    } else if(child_location->kind() == ExprKind::Unary) {
        fold_unary_constants(child_location,
                             static_cast<UnaryExpression*>(child_location.get()));
    }
}

template<typename Other>
static
void set_literal_range_type(Expression* literal, const Other* other,
                            const RangeType* other_type)
{
    const IntRange& range = other_type->range;
    switch(literal->kind()) {
    case ExprKind::IntLiteral: {
        auto* int_literal = static_cast<IntLiteral*>(literal);
        if(!range.contains(int_literal->value)) {
            Error(int_literal->line_num())
                .put("Integer Literal ").put(int_literal)
                .put(" is not in the range of:\n  ")
                .put(other_type).newline()
                .put(" so it cannot be used with:\n  ").put(other).raise();
        }
        // Literals (if used with a compatible range type) take on the type of what
        // they are used with
        int_literal->actual_type = other_type;
        break;
    }
    case ExprKind::CharLiteral: {
        auto* char_literal = static_cast<CharLiteral*>(literal);
        if(other_type != &RangeType::Character) {
            // TODO: allow user-defined character types to be used with character
            // literals
            Error(char_literal->line_num())
                .put("Character Literal ").put(char_literal)
                .put(" cannot be used with the non-character type:\n  ")
                .put(other_type).newline()
                .put("so it cannot be used with:\n  ").put(other).raise();
        }
        char_literal->actual_type = other_type;
        break;
    }
    default:
        Error(literal->line_num())
            .put("Expected a range/char literal, but instead found expression:\n  ")
            .put(literal).put("\t").put(literal->type()).raise();
    }
}

static
void set_literal_bool_type(Expression* literal, const Type* other_type)
{
    // TODO: add support for user-created boolean types, which should be usable
    // with Boolean literals
    if(literal->kind() == ExprKind::BoolLiteral) {
        auto* bool_literal = static_cast<BoolLiteral*>(literal);
        bool_literal->actual_type = other_type;
    } else {
        Error(literal->line_num())
            .put("Expected a boolean literal, but instead found expression:\n  ")
            .put(literal).put("\t").put(literal->type()).raise();
    }
}

static
void set_literal_array_type(Expression*, const ArrayType*);

template<typename Other>
static
void set_literal_type(Expression* literal, const Other* other, const Type* other_type)
{
    if(literal->type()->kind() != TypeKind::Literal)
        return;

    switch(other_type->kind()) {
    case TypeKind::Range:
        set_literal_range_type(literal, other, static_cast<const RangeType*>(other_type));
        break;
    case TypeKind::Boolean:
        set_literal_bool_type(literal, other_type);
        break;
    case TypeKind::Array:
        set_literal_array_type(literal, static_cast<const ArrayType*>(other_type));
        break;
    case TypeKind::Ref:
        // References can be implicitly dereferenced when used in expressions.
        // Since literals are always values, not references, try to resolve their
        // type to the dereferenced type of the reference
        set_literal_type(literal, other,
                         static_cast<const RefType*>(other_type)->inner_type);
        break;
    case TypeKind::Literal:
        // Ignore case of two Literal operands; will get constant folded
        break;
    case TypeKind::Normal:
        assert(false);
        break;
    }
}

static
void set_literal_array_type(Expression* literal, const ArrayType* other_type)
{
    if(literal->kind() == ExprKind::InitList) {
        auto* init_list = static_cast<InitList*>(literal);
        init_list->actual_type = other_type;
        for(auto& expr : init_list->values) {
            set_literal_type(expr.get(), init_list, other_type->element_type);
        }
    } else {
        Error(literal->line_num())
            .put("Expected an initializer list, but instead found expression:\n  ")
            .put(literal).put("\t").put(literal->type()).raise();
    }
}

void UnaryExpression::cleanup()
{
    fold_constants(right);
}

void BinaryExpression::cleanup()
{
    fold_constants(left);
    fold_constants(right);
    set_literal_type(left.get(), right.get(), right->type());
    set_literal_type(right.get(), left.get(), left->type());
}

void FunctionCall::cleanup()
{
    if(definition->parameters.size() != arguments.size()) {
        Error(line_num()).put("Function").quote(name()).put("expects ")
            .put(definition->parameters.size()).put(" arguments, but ")
            .put(arguments.size()).raise(" were provided");
    }
    auto param = definition->parameters.begin();
    for(auto& arg : arguments) {
        fold_constants(arg);
        set_literal_type(arg.get(), param->get(), (*param)->type);
        ++param;
    }
}

void IndexOp::cleanup()
{
    // Base expression will not consist of anything that can be simplified down
    // to a single literal (because it is some sort of variable usage), so no need
    // to try and fold it; instead, just call cleanup() so any of its subexpressions
    // are folded if possible.
    base_expr->cleanup();
    fold_constants(index_expr);

    const auto* base_type = base_expr->type();
    if(base_type->kind() != TypeKind::Array) {
        Error(line_num()).put(" Object\n ")
            .put(base_expr.get()).put("\t").put(base_type).newline()
            .raise(" is not an array type and so cannot be indexed using `[ ]`");
    }
    auto* array_type = static_cast<const ArrayType*>(base_type);
    set_literal_type(index_expr.get(), base_expr.get(), array_type->index_type);
}

void InitList::cleanup()
{
    for(auto& val : values) {
        fold_constants(val);
    }
}

void BasicStatement::cleanup(Cleanup&)
{
    fold_constants(expression);
}

void Initialization::cleanup(Cleanup&)
{
    if(expression != nullptr) {
        fold_constants(expression);
        set_literal_type(expression.get(), variable.get(), variable->type);
    }
}

void Assignment::cleanup(Cleanup&)
{
    fold_constants(expression);
    set_literal_type(expression.get(), assignable, assignable->type);
}

void Block::cleanup(Cleanup& context)
{
    for(auto& stmt : statements) {
        stmt->cleanup(context);
    }
}

void IfBlock::cleanup(Cleanup& context)
{
    fold_constants(condition);
    Block::cleanup(context);
    if(else_or_else_if != nullptr) {
        else_or_else_if->cleanup(context);
    }
}

void WhileLoop::cleanup(Cleanup& context)
{
    fold_constants(condition);
    Block::cleanup(context);
}

void ReturnStatement::cleanup(Cleanup& context)
{
    if(expression != nullptr) {
        fold_constants(expression);
        set_literal_type(expression.get(), context.m_curr_funct,
                         context.m_curr_funct->return_type);
    }
}

void Cleanup::run()
{
    for(Magnum::Pointer<Initialization>& var : m_global_vars) {
        var->cleanup(*this);
    }
    for(auto& funct : m_functions) {
        if(funct->kind() == FunctionKind::Normal) {
            m_curr_funct = static_cast<BBFunction*>(funct.get());
            m_curr_funct->body.cleanup(*this);
        }
    }
}
