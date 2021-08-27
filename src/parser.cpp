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
#include "error.h"
#include "parser.h"
#include "constanteval.h"
#include <ostream>
#include <algorithm>
#include <array>
#include <cassert>

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
    op_table[size_t(TokenType::Op_Xor)] = 6;
    op_table[size_t(TokenType::Op_Or)] = 5;
    // Comparison
    op_table[size_t(TokenType::Op_Eq)] = 11;
    op_table[size_t(TokenType::Op_Ne)] = 11;
    op_table[size_t(TokenType::Op_Lt)] = 12;
    op_table[size_t(TokenType::Op_Gt)] = 12;
    op_table[size_t(TokenType::Op_Le)] = 12;
    op_table[size_t(TokenType::Op_Ge)] = 12;
    op_table[size_t(TokenType::Op_Left_Shift)] = 13;
    op_table[size_t(TokenType::Op_Right_Shift)] = 13;
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

static void check_token_is(TokenType type, const char* description, Token token)
{
    if(token.type != type) {
        raise_error_expected(description, token);
    }
}

// Constructs a new object inside the given list of smart pointers, returning
// a non-owning pointer to the new object
template<typename T, typename ListT, typename ...Params>
static ListT* create(std::vector<Magnum::Pointer<ListT>>& type_list, Params... params)
{
    return type_list.emplace_back(
        Magnum::pointer<T>(std::forward<Params>(params)...)).get();
}

Parser::Parser(TokenIterator input_begin, TokenIterator input_end,
               std::vector<Magnum::Pointer<Function>>& functions,
               std::vector<Magnum::Pointer<Type>>& types,
               std::vector<Magnum::Pointer<Initialization>>& global_vars,
               std::vector<Magnum::Pointer<IndexedVariable>>& index_vars)
    : m_input_begin(input_begin), m_input_end(input_end), m_functions(functions),
      m_types(types), m_global_vars(global_vars), m_index_vars(index_vars)
{
    m_names_table.add_type(&RangeType::Integer);
    m_names_table.add_type(&RangeType::Character);
    m_names_table.add_type(&EnumType::Boolean);

    {
        // function putchar(c : Character): Integer
        auto* put_char = new BuiltinFunction("putchar");
        auto* c = new Variable("c", &RangeType::Character);
        put_char->parameters.emplace_back(c);
        put_char->return_type = &RangeType::Integer;
        m_names_table.add_function(put_char);
        m_functions.emplace_back(put_char);
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
        left_side = Magnum::pointer<BinaryExpression>(std::move(left_side), op->type,
                                                      parse_expression(op->type));
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

Magnum::Pointer<Expression> Parser::in_var_expression()
{
    const TokenIterator current = token++;
    check_token_is(TokenType::Name, "variable/constant name", *current);

    // Need to find the right Variable that this VariableExpression is a usage of
    const auto match = m_names_table.find(current->text);
    if(!match) {
        Error(current->line_num)
            .put("Unknown variable/constant").quote(current->text).raise();
    } else if(match.value().kind != NameType::Variable) {
        Error(current->line_num)
            .put("Expected name of variable/constant, but").quote(current->text)
            .raise("is already being used as a name");
    } else {
        return Magnum::pointer<VariableExpression>(current->line_num, match.value().variable);
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
            return in_var_expression();
        }
    case TokenType::Open_Parentheses:
        return in_parentheses();
    case TokenType::Open_Curly:
        return in_init_list();
    // Handle unary operators (not, -, to_val, to_ptr)
    case TokenType::Op_To_Val:
    case TokenType::Op_To_Ptr:
    case TokenType::Op_Not:
    case TokenType::Op_Minus: {
        auto op = token++;
        return Magnum::pointer<UnaryExpression>(op->type, in_expression());
    }
    default:
        return in_literal();
    }
}

Magnum::Pointer<Expression> Parser::in_function_call()
{
    check_token_is(TokenType::Name, "function name", *token);

    // First, assign the function call's name
    Magnum::Pointer<FunctionCall> new_function_call;
    //auto new_function_call = Magnum::pointer<FunctionCall>(token->line_num, token->text);
    const auto match = m_names_table.find(token->text);
    if(!match) {
        // If the function hasn't been declared yet, add it provisionally to name table
        // to be filled in (hopefully) later
        auto* temp_definition = create<BBFunction>(m_temp_functions, token->text);
        new_function_call = Magnum::pointer<FunctionCall>(token->line_num, temp_definition);
        m_names_table.add_unresolved(new_function_call.get());
    } else {
        const SymbolInfo& match_value = match.value();
        switch(match_value.kind) {
        case NameType::DeclaredFunct:
            // A temp declaration (one without definition) has been declared, but haven't
            // resolved its definition yet
            new_function_call = Magnum::pointer<FunctionCall>(token->line_num,
                                                              match_value.function);
            m_names_table.add_unresolved(new_function_call.get());
            break;
        case NameType::Funct:
            // Found a function with a full definition
            new_function_call = Magnum::pointer<FunctionCall>(token->line_num,
                                                              match_value.function);
            if(match_value.function->kind() == FunctionKind::Builtin) {
                auto* builtin = static_cast<BuiltinFunction*>(match_value.function);
                builtin->is_used = true;
            }
            break;
        default:
            Error(token->line_num)
                .put("Expected").quote(token->text)
                .raise("to be a function name, but it is defined as another kind of name");
        }
    }

    ++token;
    check_token_is(TokenType::Open_Parentheses,
                   "'(' token before function call argument list", *token);

    // Next, parse the arguments, each of which should be some sort of expression
    ++token;
    size_t arg_count = 0;
    while(token < m_input_end) {
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
    Error(token->line_num).raise("Function call definition ended early");
}

Magnum::Pointer<Expression> Parser::in_index_op()
{
    const unsigned int line = token->line_num;
    // For now, this will be a single VariableExpression, but in future will want
    // to support anonymous objects. Maybe one way to implement would be to parse
    // a single expression (non-operator-containing) in in_expression(), then
    // check if `[`, `(`, etc. as the next token, calling the correct parsing
    // function based on this next token and using the expression as its left side
    check_token_is(TokenType::Name, "name of array variable/constant", *token);
    Magnum::Pointer<Expression> base_expr{in_var_expression()};

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
    while(token < m_input_end) {
        if(token->type == TokenType::Closed_Curly) {
            ++token;
            return new_init_list;
        } else if(token->type == TokenType::Comma) {
            ++token;
        } else {
            new_init_list->values.emplace_back(parse_expression(TokenType::Comma));
        }
    }
    Error(token->line_num).raise("Initializer list definition ended early");
}

// Creates Variable, but does not add to symbol table
Magnum::Pointer<Variable> Parser::in_var_declaration()
{
    check_token_is(TokenType::Name, "the name of a variable", *token);
    if(auto name_exists = m_names_table.find(token->text); name_exists) {
        Error(token->line_num)
            .quote(token->text)
            .raise("cannot be used as a variable name. It is already defined as a name");
    }
    auto new_var = Magnum::pointer<Variable>(token->text);

    ++token;
    check_token_is(TokenType::Type_Indicator, "`:` before typename", *token);

    ++token;
    if(token->type == TokenType::Keyword_Const) {
        // `constant` keyword marks kind of variable
        new_var->is_mutable = false;
        ++token;
    }
    check_token_is(TokenType::Name, "typename", *token);

    const auto match = m_names_table.find(token->text);
    if(!match) {
        // If the type hasn't been declared yet, add it provisionally to name table
        // to be filled in (hopefully) later
        new_var->type = create<RangeType>(m_types, token->text);
        m_names_table.add_unresolved(new_var.get());
    } else {
        switch(match.value().kind) {
        case NameType::DeclaredType:
            // A temp type (one that lacks a definition) has already been declared, but
            // the actual definition of it hasn't been resolved yet
            new_var->type = match.value().type;
            m_names_table.add_unresolved(new_var.get());
            break;
        case NameType::Type:
            new_var->type = match.value().type;
            break;
        default:
            Error(token->line_num)
                .put("Expected").quote(token->text)
                .raise("to be a typename, but it is defined as another kind of name");
        }
    }

    return new_var;
}

Magnum::Pointer<Initialization> Parser::in_initialization()
{
    check_token_is(TokenType::Keyword_Let, "keyword `let`", *token);
    unsigned int line = token->line_num;
    ++token;
    auto new_statement = Magnum::pointer<Initialization>(line, in_var_declaration());

    ++token;
    if(token->type == TokenType::End_Statement) {
        // Declaration is valid, just no initial value was set
        ++token;
    } else if(token->type != TokenType::Op_Assign) {
        raise_error_expected("assignment operator or ';'", *token);
    } else {
        ++token;
        new_statement->expression = parse_expression();
        check_token_is(TokenType::End_Statement, "end statement (a.k.a ;)", *token);
        ++token;
    }
    m_names_table.add_var(new_statement->variable.get());
    return new_statement;
}

Magnum::Pointer<Assignment> Parser::in_assignment()
{
    auto match = m_names_table.find(token->text);
    if(!match) {
        Error(token->line_num)
            .quote(token->text)
            .raise("is not a variable name and so cannot be assigned to");
    } else if(match->kind != NameType::Variable) {
        Error(token->line_num)
            .put("Expected").quote(token->text)
            .raise("to be a variable, but it is defined as another kind of name");
    } else if(!match->variable->is_mutable) {
        Error(token->line_num)
            .put("Cannot assign to constant:").quote(token->text).raise();
    }
    Assignable* assgn_target = match->variable;

    if(std::next(token)->type == TokenType::Open_Bracket) {
        // Array element assignment
        Magnum::Pointer<Expression> index_op{parse_expression()};
        if(index_op->kind() != ExprKind::IndexOp) {
            Error(index_op->line_num()).raise("Expected an assignment to an array element");
        }
        auto* index_op_act = static_cast<IndexOp*>(index_op.release());
        assgn_target = create<IndexedVariable>(m_index_vars,
                                               Magnum::Pointer<IndexOp>{index_op_act});
        assert(assgn_target != nullptr);
    } else {
        // Named variable assignment; skip the varname
        ++token;
    }
    check_token_is(TokenType::Op_Assign, "assignment operator", *token);
    ++token;
    Magnum::Pointer<Expression> expr{parse_expression()};
    check_token_is(TokenType::End_Statement, "end of statement (a.k.a. `;`)", *token);
    ++token;
    return Magnum::pointer<Assignment>(std::move(expr), assgn_target);
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
        for(auto lookahead = std::next(token);
            lookahead->type != TokenType::End_Statement; ++lookahead) {
            if(lookahead->type == TokenType::Op_Assign) {
                return in_assignment();
            }
        }
        return in_basic_statement();
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
    while(token < m_input_end) {
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
    Error(token->line_num).raise("Incomplete if-block");
}

Magnum::Pointer<Block> Parser::in_else_block()
{
    check_token_is(TokenType::Keyword_Else, "keyword `else`", *token);
    m_names_table.open_scope();
    auto new_else_block = Magnum::pointer<Block>(token->line_num);
    ++token;

    while(token < m_input_end) {
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
    Error(token->line_num).raise("Incomplete else-block");
}

Magnum::Pointer<WhileLoop> Parser::in_while_loop()
{
    check_token_is(TokenType::Keyword_While, "keyword `while`", *token);
    ++token;
    m_names_table.open_scope();
    // First, parse the condition
    auto new_while_loop = Magnum::pointer<WhileLoop>(parse_expression());
    check_token_is(TokenType::Keyword_Loop,
                   "keyword `loop` following `while` condition", *token);

    // Parse the statements inside the loop body
    ++token;
    while(token < m_input_end) {
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
    Error(token->line_num).raise("Incomplete while-loop-block");
}

Magnum::Pointer<ReturnStatement> Parser::in_return_statement()
{
    check_token_is(TokenType::Keyword_Return, "`return` keyword", *token);
    auto return_stmt = Magnum::pointer<ReturnStatement>(token->line_num);
    ++token;

    if(token->type != TokenType::End_Statement) {
        return_stmt = Magnum::pointer<ReturnStatement>(parse_expression());
        check_token_is(TokenType::End_Statement, "end of statement (a.k.a. `;`)",
                   *token);
    }
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
            Error(token->line_num)
                .put("Expected").quote(token->text)
                .raise("to be a typename, but it is defined as another kind of name");
        }
    } else {
        // Type wasn't declared yet; add provisionally to name table
        funct->return_type = create<RangeType>(m_types, token->text);
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
        Error(token->line_num)
            .put("Name").quote(token->text).raise("is already in use");
    }
    auto* new_funct = new BBFunction(token->text, token->line_num);
    m_functions.emplace_back(new_funct);
    m_names_table.add_function(new_funct);
    ++token;
    check_token_is(TokenType::Open_Parentheses,
                   "`(` to follow name of function in definition", *token);

    m_names_table.open_scope();
    // Next, parse the parameters
    ++token;
    while(token < m_input_end) {
        if(token->type == TokenType::Name) {
            // Add a new parameter declaration
            Variable* param = new_funct->parameters.emplace_back(in_var_declaration()).get();
            m_names_table.add_var(param);

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
    while(token < m_input_end) {
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
    if(expr->kind() != ExprKind::Binary) {
        raise_error_expected("binary expression with operator `thru` or `upto`", *token);
    }
    auto* range_expr = static_cast<BinaryExpression*>(expr.get());
    if(range_expr->op != TokenType::Op_Thru && range_expr->op != TokenType::Op_Upto) {
        raise_error_expected("binary expression with operator `thru` or `upto`", *token);
    }

    fold_constants(range_expr->left);
    fold_constants(range_expr->right);
    if(range_expr->left->kind() != ExprKind::IntLiteral) {
        raise_error_expected("integer constant expression as the range's lower bound",
                             range_expr->left.get());
    } else if(range_expr->right->kind() != ExprKind::IntLiteral) {
        raise_error_expected("integer constant expression as the range's upper bound",
                             range_expr->right.get());
    }
    auto* left_expr = static_cast<const IntLiteral*>(range_expr->left.get());
    auto* right_expr = static_cast<const IntLiteral*>(range_expr->right.get());

    // TODO: Support arbitrary expressions made of arithmetic
    //  operators/parentheses/negations/bitwise operators
    if(right_expr->value < left_expr->value) {
        Error(token->line_num)
            .raise("Upper limit of range is lower than the lower limit");
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

    Type* new_type = create<RangeType>(m_types, type_name, lower_limit, upper_limit);
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
    if(!match || match.value().kind != NameType::Type) {
        // TODO: support unresolved array element types
        raise_error_expected("defined type", *token);
    } else if(match.value().type->kind() == TypeKind::Ref) {
        Error(token->line_num).raise("reference types cannot be elements in an array");
    }
    ++token;
    Type* new_type = create<ArrayType>(m_types, type_name, index_type, match.value().type);
    m_names_table.add_type(new_type);
}

template<typename T>
void Parser::in_ptr_like_type_definition(const std::string& type_name)
{
    static_assert(std::is_base_of_v<PtrLikeType, T>);
    check_token_is(TokenType::Name, "typename", *token);
    auto match = m_names_table.find(token->text);
    if(!match || match.value().kind != NameType::Type) {
        raise_error_expected("name of a type", *token);
    }
    ++token;

    Type* new_type = create<T>(m_types, type_name, match.value().type);
    m_names_table.add_type(new_type);
}

void Parser::in_type_definition()
{
    check_token_is(TokenType::Keyword_Type, "keyword `type`", *token);
    ++token;
    check_token_is(TokenType::Name, "typename", *token);

    if(const auto match = m_names_table.find(token->text);
       match && match.value().kind != NameType::DeclaredType) {
        Error(token->line_num)
            .put("Name:").quote(token->text).raise("is already in use");
    }

    const auto& type_name = token->text;
    ++token;
    check_token_is(TokenType::Keyword_Is, "keyword `is`", *token);

    // TODO: add ability to distinguish between new distinct types and new subtypes
    ++token;
    switch(token->type) {
    case TokenType::Keyword_Range:
        // Handle discrete types
        ++token; // Eat keyword `range`
        in_range_type_definition(type_name);
        break;
    case TokenType::Open_Bracket:
        ++token;
        in_array_type_definition(type_name);
        break;
    case TokenType::Keyword_Ref:
        ++token;
        in_ptr_like_type_definition<RefType>(type_name);
        break;
    case TokenType::Keyword_Ptr:
        ++token;
        in_ptr_like_type_definition<PtrType>(type_name);
        break;
    default:
        // TODO: handle record types, etc. here
        raise_error_expected("type definition", *token);
    }

    check_token_is(TokenType::End_Statement, "end of statement (a.k.a. `;`)", *token);
    ++token;
}

void Parser::run()
{
    token = m_input_begin;
    while(token < m_input_end) {
        switch(token->type) {
        case TokenType::Keyword_Funct:
            in_function_definition();
            break;
        case TokenType::Keyword_Type:
            in_type_definition();
            break;
        case TokenType::Keyword_Let:
            m_global_vars.push_back(in_initialization());
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
    for(const auto& decl : parser.m_global_vars) {
        decl->print(output);
    }
    output << "\nFunctions:\n";
    for(const auto& function_definition : parser.m_functions) {
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

void SymbolTable::add_var(Variable* var)
{
    m_scopes[m_curr_scope].symbols[var->name] = SymbolInfo{NameType::Variable, var};
}

void SymbolTable::add_type(Type* type)
{
    m_scopes[m_curr_scope].symbols[type->name] = SymbolInfo{NameType::Type, type};
}

void SymbolTable::add_function(Function* function)
{
    m_scopes[m_curr_scope].symbols[function->name] = SymbolInfo{NameType::Funct, function};
}

void SymbolTable::add_unresolved(Variable* var)
{
    m_scopes[m_curr_scope].unresolved_types.push_back(&var->type);
}

void SymbolTable::add_unresolved(FunctionCall* funct)
{
    m_scopes[m_curr_scope].unresolved_funct_calls.push_back(funct);
}

void SymbolTable::add_unresolved_return_type(Function* funct)
{
    m_scopes[m_curr_scope].unresolved_types.push_back(&funct->return_type);
}

void SymbolTable::resolve_usages()
{
    // Check that all functions and types in this module have a definition.
    // If they don't, try to find one/fix-up the symbol table

    // TODO: Have mechanism where types/functions in other modules are resolved
    const int prev_scope = m_curr_scope;
    for(size_t scope_index = 0; scope_index < m_scopes.size(); ++scope_index) {
        m_curr_scope = scope_index;
        Scope& scope = m_scopes[scope_index];
        // Try and resolve the references to types that were missing a definition
        // when first used
        for(Type** type_ptr : scope.unresolved_types) {
            std::optional<SymbolInfo> match = find((*type_ptr)->name, NameType::Type);
            if(!match) {
                Error().put("Type").quote((*type_ptr)->name)
                    .raise("is used but has no definition");
            } else {
                // Update to the newly-defined type (replacing the temp type)
                *type_ptr = match->type;
            }
        }
        scope.unresolved_types.clear();

        // Try and resolve function calls to functions declared later
        for(FunctionCall* funct_call : scope.unresolved_funct_calls) {
            std::optional<SymbolInfo> match = find(funct_call->name(), NameType::Funct);
            if(!match) {
                Error().put("Function").quote(funct_call->name())
                    .raise("is used but has no definition");
            } else {
                // Update call to point to the actual function
                funct_call->definition = match->function;
            }
        }
        scope.unresolved_funct_calls.clear();
    }
    // Restore original
    m_curr_scope = prev_scope;
}
