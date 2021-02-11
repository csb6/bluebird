/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2021  Cole Blakley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include "parser.h"
#include <string_view>
#include <iostream>
#include <algorithm>
#include <array>

using Precedence = signed char;
constexpr Precedence Invalid_Binary_Operator = -2;
constexpr Precedence Operand = 100;

// Only contains precedences for binary operators
constexpr auto operator_precedence_table = []()
{
    constexpr size_t table_size = size_t(TokenType::Name) + 1;
    std::array<Precedence, table_size> op_table{};

    for(auto& precedence : op_table) {
        precedence = Invalid_Binary_Operator;
    }
    // Arithmetic
    op_table[size_t(TokenType::Op_Plus)] = 14;
    op_table[size_t(TokenType::Op_Minus)] = 14;
    op_table[size_t(TokenType::Op_Div)] = 15;
    op_table[size_t(TokenType::Op_Mult)] = 15;
    op_table[size_t(TokenType::Op_Mod)] = 15;
    op_table[size_t(TokenType::Op_Rem)] = 15;
    // Logical
    op_table[size_t(TokenType::Op_And)] = 7;
    op_table[size_t(TokenType::Op_Or)] = 6;
    // Comparison
    op_table[size_t(TokenType::Op_Eq)] = 11;
    op_table[size_t(TokenType::Op_Ne)] = 11;
    op_table[size_t(TokenType::Op_Lt)] = 12;
    op_table[size_t(TokenType::Op_Gt)] = 12;
    op_table[size_t(TokenType::Op_Le)] = 12;
    op_table[size_t(TokenType::Op_Ge)] = 12;
    op_table[size_t(TokenType::Op_Left_Shift)] = 13;
    op_table[size_t(TokenType::Op_Right_Shift)] = 13;
    op_table[size_t(TokenType::Op_Bit_And)] = 10;
    op_table[size_t(TokenType::Op_Bit_Or)] = 8;
    op_table[size_t(TokenType::Op_Bit_Xor)] = 9;
    // Range
    op_table[size_t(TokenType::Op_Thru)] = 13;
    op_table[size_t(TokenType::Op_Upto)] = 13;
    // Operands
    op_table[size_t(TokenType::String_Literal)] = Operand;
    op_table[size_t(TokenType::Char_Literal)] = Operand;
    op_table[size_t(TokenType::Int_Literal)] = Operand;
    op_table[size_t(TokenType::Float_Literal)] = Operand;
    op_table[size_t(TokenType::Name)] = Operand;

    return op_table;
}();

constexpr Precedence precedence_of(const TokenType index)
{
    return operator_precedence_table[(unsigned char)index];
}

constexpr bool is_binary_operator(const Precedence p)
{
    return p >= 0 && p != Operand;
}


[[noreturn]] static void raise_error(unsigned int line_num, std::string_view message)
{
    std::cerr << "ERROR: Line " << line_num << ": "
              << message << "\n";
    exit(1);
}

[[noreturn]] static void raise_error(std::string_view message)
{
    std::cerr << "ERROR: " << message << "\n";
    exit(1);
}

[[noreturn]] static void raise_error_expected(std::string_view expected, Token actual)
{
    std::cerr << "ERROR: Line " << actual.line_num << ": "
              << "Expected " << expected << ", but instead found token:\n";
    std::cerr << "  " << actual;
    exit(1);
}

[[noreturn]] static void raise_error_expected(std::string_view expected, const Expression* actual)
{
    std::cerr << "ERROR: Line " << actual->line_num() << ": "
              << "Expected " << expected << ", but instead found expression:\n";
    actual->print(std::cerr);
    std::cerr << "\n";
    exit(1);
}

static void check_token_is(TokenType type, std::string_view description, Token token)
{
    if(token.type != type) {
        raise_error_expected(description, token);
        exit(1);
    }
}

static Magnum::Pointer<Expression>
fold_constants(Token op, Magnum::Pointer<Expression>&& right)
{
    switch(right->kind()) {
    case ExpressionKind::IntLiteral: {
        auto* r_int = static_cast<IntLiteral*>(right.get());
        switch(op.type) {
        case TokenType::Op_Minus:
            r_int->value.negate();
            break;
        case TokenType::Op_Bit_Not:
            r_int->value.ones_complement();
            break;
        default:
            raise_error_expected("unary operator that works on integer literals", op);
        }
        return std::move(right);
    }
    case ExpressionKind::BoolLiteral: {
        auto* r_bool = static_cast<BoolLiteral*>(right.get());
        if(op.type == TokenType::Op_Not) {
            r_bool->value = !r_bool->value;
        } else {
            raise_error_expected("logical unary operator (e.g. `not`)", op);
        }
        return std::move(right);
    }
    default:
        return Magnum::pointer<UnaryExpression>(op.type, std::move(right));
    }
}

static Magnum::Pointer<Expression>
fold_constants(Magnum::Pointer<Expression>&& left, const Token& op,
               Magnum::Pointer<Expression>&& right)
{
    const auto left_kind = left->kind();
    const auto right_kind = right->kind();
    if(left_kind == ExpressionKind::IntLiteral && right_kind == ExpressionKind::IntLiteral) {
        // Fold into a single literal (uses arbitrary-precision arithmetic)
        auto* l_int = static_cast<IntLiteral*>(left.get());
        auto* r_int = static_cast<IntLiteral*>(right.get());
        switch(op.type) {
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
        case TokenType::Op_Bit_And:
            l_int->value &= r_int->value;
            break;
        case TokenType::Op_Bit_Or:
            l_int->value |= r_int->value;
            break;
        case TokenType::Op_Bit_Xor:
            l_int->value ^= r_int->value;
            break;
        case TokenType::Op_Thru:
        case TokenType::Op_Upto:
            // Can't do folds here, need to preserve left/right sides for a range
            return Magnum::pointer<BinaryExpression>(std::move(left), op.type,
                                                     std::move(right));
        default:
            raise_error_expected("binary operator that works on integer literals", op);
        }
        return std::move(left);
    } else if(left_kind == ExpressionKind::BoolLiteral
              && right_kind == ExpressionKind::BoolLiteral) {
        // Fold into a single literal
        auto* l_bool = static_cast<BoolLiteral*>(left.get());
        auto* r_bool = static_cast<BoolLiteral*>(right.get());
        switch(op.type) {
        case TokenType::Op_And:
            l_bool->value &= r_bool->value;
            break;
        case TokenType::Op_Or:
            l_bool->value |= r_bool->value;
            break;
        default:
            raise_error_expected("logical binary operator", op);
        }
        return std::move(left);
    }
    return Magnum::pointer<BinaryExpression>(std::move(left), op.type,
                                             std::move(right));
}

// Constructs a new object inside the given list of smart pointers, returning
// a non-owning pointer to the new object
template<typename T, typename ListT, typename ...Params>
static ListT* create(std::vector<Magnum::Pointer<ListT>>& type_list, Params... params)
{
    return type_list.emplace_back(
        Magnum::pointer<T>(std::forward<Params>(params)...)).get();
}

Parser::Parser(TokenIterator input_begin, TokenIterator input_end)
    : m_input_begin(input_begin), m_input_end(input_end)
{
    m_names_table.add_type(&RangeType::Integer);
    m_names_table.add_type(&RangeType::Character);
    m_names_table.add_type(&EnumType::Boolean);

    {
        // function putchar(c : Character): Integer
        auto* put_char = new BuiltinFunction("putchar");
        auto* c = new LValue("c", &RangeType::Character);
        put_char->parameters.emplace_back(c);
        put_char->return_type = &RangeType::Integer;
        m_names_table.add_function(put_char);
        m_function_list.emplace_back(put_char);
    }
}

// Pratt parser
Magnum::Pointer<Expression> Parser::parse_expression(TokenType right_token)
{
    Magnum::Pointer<Expression> left_side{in_expression()};
    const Precedence right_precedence = precedence_of(right_token);
    Precedence curr_precedence = precedence_of(token->type);
    while(right_precedence < curr_precedence && token->type != TokenType::End_Statement) {
        auto op = token;
        if(!is_binary_operator(curr_precedence)) {
            raise_error_expected("binary operator", *token);
        }
        ++token;
        left_side = fold_constants(std::move(left_side), *op, parse_expression(op->type));
        curr_precedence = precedence_of(token->type);
    }
    return left_side;
}

Magnum::Pointer<Expression> Parser::in_literal()
{
    const TokenIterator current = token++;

    try {
        switch(current->type) {
        case TokenType::String_Literal:
            return Magnum::pointer<StringLiteral>(current->line_num, current->text);
        case TokenType::Char_Literal:
            return Magnum::pointer<CharLiteral>(current->line_num, current->text[0]);
        case TokenType::Int_Literal:
            return Magnum::pointer<IntLiteral>(current->line_num, current->text);
        case TokenType::Float_Literal:
            return Magnum::pointer<FloatLiteral>(current->line_num, std::stod(current->text));
        case TokenType::Keyword_True:
            return Magnum::pointer<BoolLiteral>(current->line_num, true);
        case TokenType::Keyword_False:
            return Magnum::pointer<BoolLiteral>(current->line_num, false);
        default:
            raise_error_expected("literal", *current);
        }
    } catch(const std::invalid_argument&) {
        raise_error_expected("float literal", *current);
    } catch(const std::out_of_range&) {
        raise_error_expected("float literal", *current);
    }
}

Magnum::Pointer<Expression> Parser::in_lvalue_expression()
{
    const TokenIterator current = token++;
    check_token_is(TokenType::Name, "variable/constant name", *current);

    // Need to find the right LValue that this LValueExpression is a usage of
    const auto match = m_names_table.find(current->text);
    if(!match) {
        raise_error(current->line_num,
                    "Unknown variable/constant `" + current->text + "`");
    } else if(match.value().kind != NameType::LValue) {
        raise_error("Expected name of variable/constant, but `"
                    + current->text + "` is already being used as a name");
    } else {
        return Magnum::pointer<LValueExpression>(
            current->line_num, current->text, match.value().lvalue);
    }
}

Magnum::Pointer<Expression> Parser::in_expression()
{
    switch(token->type) {
    case TokenType::Name:
        if(std::next(token)->type == TokenType::Open_Parentheses) {
            return in_function_call();
        } else if(std::next(token)->type == TokenType::Open_Bracket) {
            return in_index_op();
        } else {
            return in_lvalue_expression();
        }
    case TokenType::Open_Parentheses:
        return in_parentheses();
    case TokenType::Open_Curly:
        return in_init_list();
    // Handle unary operators (not, ~, -)
    case TokenType::Op_Not:
    case TokenType::Op_Bit_Not:
    case TokenType::Op_Minus: {
        auto op = token;
        ++token;
        return fold_constants(*op, in_expression());
    }
    default:
        return in_literal();
    }
}

Magnum::Pointer<Expression> Parser::in_function_call()
{
    check_token_is(TokenType::Name, "function name", *token);

    // First, assign the function call's name
    auto new_function_call = Magnum::pointer<FunctionCall>(token->line_num, token->text);
    const auto match = m_names_table.find(token->text);
    bool is_resolved = false;
    if(!match) {
        // If the function hasn't been declared yet, add it provisionally to name table
        // to be filled in (hopefully) later
        new_function_call->function = create<BBFunction>(m_temp_function_list, token->text);
        m_names_table.add_unresolved(new_function_call.get());
    } else {
        SymbolInfo match_value{match.value()};
        switch(match_value.kind) {
        case NameType::DeclaredFunct:
            // A temp declaration (one without definition) has been declared, but haven't
            // resolved its definition yet
            new_function_call->function = match_value.function;
            m_names_table.add_unresolved(new_function_call.get());
            break;
        case NameType::Funct:
            // Found a function with a full definition
            new_function_call->function = match_value.function;
            is_resolved = true;
            if(match_value.function->kind() == FunctionKind::Builtin) {
                auto* builtin = static_cast<BuiltinFunction*>(match_value.function);
                builtin->is_used = true;
            }
            break;
        default:
            raise_error(token->line_num, "Expected `" + token->text
                        + "` to be a function name, but it is defined as another kind of name");
        }
    }

    ++token;
    check_token_is(TokenType::Open_Parentheses,
                   "'(' token before function call argument list", *token);

    // Next, parse the arguments, each of which should be some sort of expression
    ++token;
    size_t arg_count = 0;
    while(token != m_input_end) {
        switch(token->type) {
        case TokenType::Comma:
            // The comma after an argument expression; continue
            ++token;
            break;
        case TokenType::Closed_Parentheses:
            ++token;
            return new_function_call;
        default:
            new_function_call->arguments.push_back(parse_expression(TokenType::Comma));
            ++arg_count;
        }
    }
    raise_error(token->line_num, "Function call definition ended early");
}

Magnum::Pointer<Expression> Parser::in_index_op()
{
    const unsigned int line = token->line_num;
    // For now, this will be a single LValueExpression, but in future will want
    // to support anonymous objects (and other rvalues). Maybe one way to implement
    // would be to parse a single expression (non-operator-containing) in in_expression(),
    // then check if `[`, `(`, etc. as the next token, calling the correct parsing
    // function based on this next token and using the expression as its left side
    check_token_is(TokenType::Name, "name of array variable/constant", *token);
    Magnum::Pointer<Expression> base_expr{in_lvalue_expression()};

    check_token_is(TokenType::Open_Bracket, "array open bracket `[`", *token);
    ++token;
    auto new_expr = Magnum::pointer<IndexOp>(
        line, std::move(base_expr), parse_expression(TokenType::Closed_Bracket));
    check_token_is(TokenType::Closed_Bracket, "array closing bracket `]`", *token);
    ++token;
    return new_expr;
}

Magnum::Pointer<Expression> Parser::in_parentheses()
{
    check_token_is(TokenType::Open_Parentheses,
                   "opening parentheses for an expression group", *token);

    ++token;
    Magnum::Pointer<Expression> result{parse_expression(TokenType::Closed_Parentheses)};
    if(token->type != TokenType::Closed_Parentheses) {
        raise_error_expected("closing parentheses for an expression group", *token);
    }

    ++token;
    return result;
}

Magnum::Pointer<Expression> Parser::in_init_list()
{
    check_token_is(TokenType::Open_Curly, "curly bracket", *token);
    auto new_init_list = Magnum::pointer<InitList>(token->line_num);
    ++token;
    while(token != m_input_end) {
        if(token->type == TokenType::Closed_Curly) {
            ++token;
            return new_init_list;
        } else if(token->type == TokenType::Comma) {
            ++token;
        } else {
            new_init_list->values.emplace_back(parse_expression(TokenType::Comma));
        }
    }
    raise_error(token->line_num, "Initializer list definition ended early");
}

// Creates LValue, but does not add to symbol table
Magnum::Pointer<LValue> Parser::in_lvalue_declaration()
{
    check_token_is(TokenType::Name, "the name of an lvalue", *token);
    if(auto name_exists = m_names_table.find(token->text); name_exists) {
        raise_error(token->line_num, "`" + token->text
                    + "` cannot be used as an lvalue name. It is already defined as a name");
    }
    auto new_lvalue = Magnum::pointer<LValue>(token->text);

    ++token;
    check_token_is(TokenType::Type_Indicator, "`:` before typename", *token);

    ++token;
    if(token->type == TokenType::Keyword_Const) {
        // `constant` keyword marks kind of lvalue
        new_lvalue->is_mutable = false;
        ++token;
    }
    check_token_is(TokenType::Name, "typename", *token);

    const auto match = m_names_table.find(token->text);
    if(!match) {
        // If the type hasn't been declared yet, add it provisionally to name table
        // to be filled in (hopefully) later
        new_lvalue->type = create<RangeType>(m_type_list, token->text);
        m_names_table.add_unresolved(new_lvalue.get());
    } else {
        switch(match.value().kind) {
        case NameType::DeclaredType:
            // A temp type (one that lacks a definition) has already been declared, but
            // the actual definition of it hasn't been resolved yet
            new_lvalue->type = match.value().type;
            m_names_table.add_unresolved(new_lvalue.get());
            break;
        case NameType::Type:
            new_lvalue->type = match.value().type;
            break;
        default:
            raise_error(token->line_num, "Expected `" + token->text
                        + "` to be a typename, but it is defined as another kind of name");
        }
    }

    return new_lvalue;
}

Magnum::Pointer<Initialization> Parser::in_initialization()
{
    check_token_is(TokenType::Keyword_Let, "keyword `let`", *token);
    ++token;
    auto new_statement = Magnum::pointer<Initialization>(in_lvalue_declaration());

    ++token;
    if(token->type == TokenType::End_Statement) {
        // Declaration is valid, just no initial value was set
        ++token;
    } else if(token->type != TokenType::Op_Assign) {
        raise_error_expected("assignment operator or ';'", *token);
    } else {
        ++token;
        new_statement->expression = parse_expression();
        ++token;
        if(new_statement->expression->kind() == ExpressionKind::InitList) {
            // Initialization lists need a backpointer to the lvalue they are used with
            auto* init_list = static_cast<InitList*>(new_statement->expression.get());
            init_list->lvalue = new_statement->lvalue.get();
        }
    }
    m_names_table.add_lvalue(new_statement->lvalue.get());
    return new_statement;
}

Magnum::Pointer<Assignment> Parser::in_assignment()
{
    auto lval_match = m_names_table.find(token->text);
    if(!lval_match) {
        raise_error(token->line_num, "`" + token->text
                    + "` is not a variable name and so cannot be assigned to");
    } else if(lval_match->kind != NameType::LValue) {
        raise_error(token->line_num, "Expected `" + token->text
                    + "` to be a variable, but it is defined as another kind of name");
    } else if(!lval_match->lvalue->is_mutable) {
        raise_error(token->line_num, "cannot assign to constant `" + token->text + "`");
        
    }

    std::advance(token, 2); // skip varname and `=` operator
    Magnum::Pointer<Expression> expr{parse_expression()};
    ++token;
    if(expr->kind() == ExpressionKind::InitList) {
        // Initialization lists need a backpointer to the lvalue they are used with
        auto* init_list = static_cast<InitList*>(expr.get());
        init_list->lvalue = lval_match->lvalue;
    }
    return Magnum::pointer<Assignment>(std::move(expr), lval_match->lvalue);
}

Magnum::Pointer<Statement> Parser::in_statement()
{
    switch(token->type) {
    case TokenType::Keyword_If:
        return in_if_block();
    case TokenType::Keyword_Let:
        return in_initialization();
    case TokenType::Keyword_While:
        return in_while_loop();
    case TokenType::Keyword_Return:
        return in_return_statement();
    default:
        if(std::next(token)->type == TokenType::Op_Assign) {
            return in_assignment();
        } else {
            return in_basic_statement();
        }
    };
}

Magnum::Pointer<BasicStatement> Parser::in_basic_statement()
{
    auto new_statement = Magnum::pointer<BasicStatement>(parse_expression());

    check_token_is(TokenType::End_Statement, "end of statement (a.k.a. `;`)", *token);
    ++token;
    return new_statement;
}

Magnum::Pointer<IfBlock> Parser::in_if_block()
{
    check_token_is(TokenType::Keyword_If, "keyword `if`", *token);
    ++token;

    m_names_table.open_scope();
    // First, parse the if-condition
    auto new_if_block = Magnum::pointer<IfBlock>(parse_expression());
    check_token_is(TokenType::Keyword_Do,
                   "keyword `do` following `if` condition", *token);

    // Next, parse the statements inside the if-block
    ++token;
    while(token != m_input_end) {
        if(token->type == TokenType::Keyword_End) {
            // End of block
            m_names_table.close_scope();
            if(std::next(token)->type == TokenType::Keyword_If) {
                // Optional keyword to show end of if-block
                ++token;
            }
            ++token;
            check_token_is(TokenType::End_Statement,
                           "closing `;` or an end label after end of if-block",
                           *token);
            ++token;
            return new_if_block;
        } else if(token->type == TokenType::Keyword_Else) {
            // End of if-block, start of else-if or else block
            m_names_table.close_scope();
            if(std::next(token)->type == TokenType::Keyword_If) {
                // Else-if block
                ++token;
                new_if_block->else_or_else_if = in_if_block();
            } else {
                // Else block
                new_if_block->else_or_else_if = in_else_block();
            }
            return new_if_block;
        } else {
            new_if_block->statements.push_back(in_statement());
        }
    }
    raise_error(token->line_num, "Incomplete if-block");
}

Magnum::Pointer<Block> Parser::in_else_block()
{
    check_token_is(TokenType::Keyword_Else, "keyword `else`", *token);
    m_names_table.open_scope();
    auto new_else_block = Magnum::pointer<Block>(token->line_num);
    ++token;

    while(token != m_input_end) {
        if(token->type == TokenType::Keyword_End) {
            // End of block
            m_names_table.close_scope();
            if(std::next(token)->type == TokenType::Keyword_If) {
                // Optional keyword to show end of if-block
                ++token;
            }
            ++token;
            check_token_is(TokenType::End_Statement,
                           "closing `;` or an end label after end of else-block",
                           *token);
            ++token;
            return new_else_block;
        } else {
            new_else_block->statements.push_back(in_statement());
        }
    }
    raise_error(token->line_num, "Incomplete else-block");
}

Magnum::Pointer<WhileLoop> Parser::in_while_loop()
{
    check_token_is(TokenType::Keyword_While, "keyword `while`", *token);
    ++token;
    m_names_table.open_scope();
    // First, parse the condition
    auto new_while_loop = Magnum::pointer<WhileLoop>(parse_expression());
    check_token_is(TokenType::Keyword_Do,
                   "keyword `do` following `while` condition", *token);

    // Parse the statements inside the loop body
    ++token;
    while(token != m_input_end) {
        if(token->type == TokenType::Keyword_End) {
            // End of block
            m_names_table.close_scope();
            if(std::next(token)->type == TokenType::Keyword_While) {
                // Optional keyword to show end of while-block
                ++token;
            }
            ++token;
            check_token_is(TokenType::End_Statement,
                           "closing `;` or an end label after end of while-block",
                           *token);
            ++token;
            return new_while_loop;
        } else {
            new_while_loop->statements.push_back(in_statement());
        }
    }
    raise_error(token->line_num, "Incomplete while-loop-block");
}

Magnum::Pointer<ReturnStatement> Parser::in_return_statement()
{
    check_token_is(TokenType::Keyword_Return, "`return` keyword", *token);

    ++token;
    auto return_stmt = Magnum::pointer<ReturnStatement>(parse_expression());
    check_token_is(TokenType::End_Statement, "end of statement (a.k.a. `;`)",
                   *token);
    ++token;
    return return_stmt;
}

void Parser::in_return_type(Function* funct)
{
    check_token_is(TokenType::Name, "function return type", *token);
    if(auto match = m_names_table.find(token->text); match) {
        switch(match.value().kind) {
        case NameType::DeclaredType:
            // A temp type has been declared previously, but not its definition
            funct->return_type = match.value().type;
            m_names_table.add_unresolved_return_type(funct);
            break;
        case NameType::Type:
            funct->return_type = match.value().type;
            break;
        default:
            raise_error(token->line_num, "Expected `" + token->text
                        + "` to be a typename, but it is defined as another kind of name");
        }
    } else {
        // Type wasn't declared yet; add provisionally to name table
        funct->return_type = create<RangeType>(m_type_list, token->text);
        m_names_table.add_unresolved_return_type(funct);
    }
    ++token;
}

void Parser::in_function_definition()
{
    ++token;
    check_token_is(TokenType::Name, "name to follow `function keyword`", *token);

    // First, set the function's name, making sure it isn't being used for
    // anything else
    const auto match = m_names_table.find(token->text);
    if(match && match.value().kind != NameType::DeclaredFunct) {
        raise_error(token->line_num, "Name `" + token->text + "` is"
                    " already in use");
    }
    auto* new_funct = new BBFunction(token->text, token->line_num);
    m_function_list.emplace_back(new_funct);
    m_names_table.add_function(new_funct);
    ++token;
    check_token_is(TokenType::Open_Parentheses,
                   "`(` to follow name of function in definition", *token);

    m_names_table.open_scope();
    // Next, parse the parameters
    ++token;
    while(token != m_input_end) {
        if(token->type == TokenType::Name) {
            // Add a new parameter declaration
            LValue* param = new_funct->parameters.emplace_back(
                in_lvalue_declaration()).get();
            m_names_table.add_lvalue(param);

            ++token;
            // Check for a comma separator after typename
            if(token->type == TokenType::Closed_Parentheses) {
                // Put back so next iteration can handle end of parameter list
                --token;
            } else if(token->type != TokenType::Comma) {
                raise_error_expected("comma to follow the parameter", *token);
            }
        } else if(token->type == TokenType::Closed_Parentheses) {
            ++token;
            // Optional return type
            if(token->type == TokenType::Type_Indicator) {
                ++token;
                in_return_type(new_funct);
            }
            // Next, look for `is` keyword
            check_token_is(TokenType::Keyword_Is,
                           "keyword `is` to follow parameters of the function", *token);
            break;
        } else {
            raise_error_expected("parameter name", *token);
        }

        ++token;
    }

    // Finally, parse the body of the function
    ++token;
    while(token != m_input_end) {
        if(token->type != TokenType::Keyword_End) {
            // Found a statement, parse it
            new_funct->body.statements.emplace_back(in_statement());
        } else {
            // End of function
            m_names_table.close_scope();
            if(std::next(token)->type == TokenType::Name
               && std::next(token)->text == new_funct->name) {
                // Optional end label for function; comes after `end`
                ++token;
            }
            ++token;
            check_token_is(TokenType::End_Statement,
                           "end of statement (a.k.a. `;`) or an end label", *token);
            ++token;
            return;
        }
    }
}

void Parser::in_range(multi_int& low_out, multi_int& high_out)
{
    Magnum::Pointer<Expression> expr = parse_expression();
    if(expr->kind() != ExpressionKind::Binary) {
        raise_error_expected("binary expression with operator `thru` or `upto`", *token);
    }
    auto* range_expr = static_cast<const BinaryExpression*>(expr.get());
    if(range_expr->op != TokenType::Op_Thru && range_expr->op != TokenType::Op_Upto) {
        raise_error_expected("binary expression with operator `thru` or `upto`", *token);
    }

    if(range_expr->left->kind() != ExpressionKind::IntLiteral) {
        raise_error_expected("integer constant expression as the range's lower bound",
                             range_expr->left.get());
    } else if(range_expr->right->kind() != ExpressionKind::IntLiteral) {
        raise_error_expected("integer constant expression as the range's upper bound",
                             range_expr->right.get());
    }
    auto* left_expr = static_cast<const IntLiteral*>(range_expr->left.get());
    auto* right_expr = static_cast<const IntLiteral*>(range_expr->right.get());

    // TODO: Support arbitrary expressions made of arithmetic
    //  operators/parentheses/negations/bitwise operators
    if(right_expr->value < left_expr->value) {
        raise_error(token->line_num, "Upper limit of range is lower than the lower limit");
    }
    low_out = left_expr->value;
    high_out = right_expr->value;
    if(range_expr->op == TokenType::Op_Upto) {
        high_out -= 1;
    }
}

void Parser::in_range_type_definition(const std::string& type_name)
{
    multi_int lower_limit, upper_limit;
    in_range(lower_limit, upper_limit);

    Type* new_type = create<RangeType>(m_type_list, type_name, lower_limit, upper_limit);
    m_names_table.add_type(new_type);
}

void Parser::in_array_type_definition(const std::string& type_name)
{
    // TODO: also allow for a range expression as the index type
    check_token_is(TokenType::Name, "name of the array's index type", *token);
    auto match = m_names_table.find(token->text);
    if(!match || match.value().kind != NameType::Type
       || match.value().type->kind() != TypeKind::Range) {
        raise_error_expected("name of a range type", *token);
    }
    auto* index_type = static_cast<const RangeType*>(match.value().type);
    ++token;

    check_token_is(TokenType::Closed_Bracket, "closing bracket `]`", *token);
    ++token;
    check_token_is(TokenType::Keyword_Of, "keyword `of`", *token);
    ++token;
    match = m_names_table.find(token->text);
    if(!match || (match && match.value().kind != NameType::Type)) {
        // TODO: support unresolved array element types
        raise_error_expected("defined type", *token);
    }
    ++token;
    Type* new_type = create<ArrayType>(m_type_list, type_name, index_type,
                                       match.value().type);
    m_names_table.add_type(new_type);
}

void Parser::in_type_definition()
{
    check_token_is(TokenType::Keyword_Type, "keyword `type`", *token);
    ++token;
    check_token_is(TokenType::Name, "typename", *token);

    if(const auto match = m_names_table.find(token->text);
       match && match.value().kind != NameType::DeclaredType) {
        raise_error(token->line_num, "Name: " + token->text + " already in use");
    }

    const auto& type_name = token->text;
    ++token;
    check_token_is(TokenType::Keyword_Is, "keyword `is`", *token);

    // TODO: add ability to distinguish between new distinct types and new subtypes
    ++token;
    if(token->type == TokenType::Keyword_Range) {
        // Handle discrete types
        ++token; // Eat keyword `range`
        in_range_type_definition(type_name);
    } else if(token->type == TokenType::Open_Bracket) {
        ++token;
        in_array_type_definition(type_name);
    } else {
        // TODO: handle record types, etc. here
        raise_error_expected("type definition", *token);
    }

    check_token_is(TokenType::End_Statement, "end of statement (a.k.a. `;`)", *token);
    ++token;
}

void Parser::run()
{
    token = m_input_begin;
    while(token != m_input_end) {
        switch(token->type) {
        case TokenType::Keyword_Funct:
            in_function_definition();
            break;
        case TokenType::Keyword_Type:
            in_type_definition();
            break;
        case TokenType::Keyword_Let:
            m_global_var_list.push_back(in_initialization());
            break;
        default:
            raise_error_expected("start of a variable/type declaration or a "
                                 "function definition", *token);
        }
    }

    m_names_table.resolve_usages();
}

std::ostream& operator<<(std::ostream& output, const Parser& parser)
{
    output << "Global Variables:\n";
    for(const auto& decl : parser.m_global_var_list) {
        decl->print(output);
    }
    output << "\nFunctions:\n";
    for(const auto& function_definition : parser.m_function_list) {
        function_definition->print(output);
        output << "\n";
    }
    return output;
}


SymbolTable::SymbolTable()
{
    m_scopes.reserve(20);
    // Add root scope
    m_scopes.push_back({-1});
    m_curr_scope = 0;
}

void SymbolTable::open_scope()
{
    m_scopes.push_back({m_curr_scope});
    m_curr_scope = m_scopes.size() - 1;
}

void SymbolTable::close_scope()
{
    m_curr_scope = m_scopes[m_curr_scope].parent_index;
}

std::optional<SymbolInfo> SymbolTable::find(const std::string& name) const
{
    int scope_index = m_curr_scope;
    while(scope_index >= 0) {
        const Scope& scope = m_scopes[scope_index];
        const auto match = scope.symbols.find(name);
        if(match != scope.symbols.end()) {
            return {match->second};
        } else {
            scope_index = scope.parent_index;
        }
    }
    return {};
}


std::optional<SymbolInfo> SymbolTable::find(const std::string& name, NameType kind) const
{
    int scope_index = m_curr_scope;
    while(scope_index >= 0) {
        const Scope& scope = m_scopes[scope_index];
        const auto match = scope.symbols.find(name);
        if(match != scope.symbols.end() && match->second.kind == kind) {
            return {match->second};
        } else {
            scope_index = scope.parent_index;
        }
    }
    return {};
}

void SymbolTable::add_lvalue(LValue* lval)
{
    m_scopes[m_curr_scope].symbols[lval->name] = SymbolInfo{NameType::LValue, lval};
}

void SymbolTable::add_type(Type* type)
{
    m_scopes[m_curr_scope].symbols[type->name] = SymbolInfo{NameType::Type, type};
}

void SymbolTable::add_function(Function* function)
{
    m_scopes[m_curr_scope].symbols[function->name] = SymbolInfo{NameType::Funct, function};
}

void SymbolTable::add_unresolved(LValue* lvalue)
{
    m_scopes[m_curr_scope].lvalues_type_unresolved.push_back(lvalue);
}

void SymbolTable::add_unresolved(FunctionCall* funct)
{
    m_scopes[m_curr_scope].unresolved_funct_calls.push_back(funct);
}

void SymbolTable::add_unresolved_return_type(Function* funct)
{
    m_scopes[m_curr_scope].unresolved_return_type_functs.push_back(funct);
}

void SymbolTable::resolve_usages()
{
    // Check that all functions and types in this module have a definition.
    // If they don't, try to find one/fix-up the symbol table

    // TODO: Have mechanism where types/functions in other modules are resolved
    const int old_curr_scope = m_curr_scope;
    for(size_t scope_index = 0; scope_index < m_scopes.size(); ++scope_index) {
        m_curr_scope = scope_index;
        Scope& scope = m_scopes[scope_index];
        // Try and resolve the types of lvalues whose types were not declared beforehand
        for(LValue* lvalue : scope.lvalues_type_unresolved) {
            std::optional<SymbolInfo> match = find(lvalue->type->name, NameType::Type);
            if(!match) {
                raise_error("Type `" + lvalue->type->name
                            + "` is used but has no definition");
            } else {
                // Update to the newly-defined type (replacing the temp type)
                lvalue->type = match->type;
            }
        }
        scope.lvalues_type_unresolved.clear();

        for(Function* funct : scope.unresolved_return_type_functs) {
            std::optional<SymbolInfo> match = find(funct->return_type->name, NameType::Type);
            if(!match) {
                raise_error("Return type `" + funct->return_type->name
                            + "` of function `" + funct->name
                            + "` is used but has no definition");
            } else {
                // Update to the newly-defined type (replacing the temp type)
                funct->return_type = match->type;
            }
        }
        scope.unresolved_return_type_functs.clear();

        // Try and resolve function calls to functions declared later
        for(FunctionCall* funct_call : scope.unresolved_funct_calls) {
            std::optional<SymbolInfo> match = find(funct_call->name, NameType::Funct);
            if(!match) {
                raise_error("Function `" + funct_call->name
                            + "` is used but has no definition");
            } else {
                // Update call to point to the actual function
                funct_call->function = match->function;
            }
        }
        scope.unresolved_funct_calls.clear();
    }
    // Restore original
    m_curr_scope = old_curr_scope;
}
