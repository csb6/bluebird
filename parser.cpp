/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020  Cole Blakley

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

using Precedence = char;
constexpr Precedence Invalid_Binary_Operator = -2;
constexpr Precedence Operand = 100;

constexpr Precedence operator_precedence_table[] = {
    // Keywords
    //  Keyword_Funct:
         Invalid_Binary_Operator,
    //  Keyword_Is:
         Invalid_Binary_Operator,
    // Keyword_Do:
         Invalid_Binary_Operator,
    //  Keyword_Let:
         Invalid_Binary_Operator,
    //  Keyword_Const:
         Invalid_Binary_Operator,
    //  Keyword_Type:
         Invalid_Binary_Operator,
    //  Keyword_End:
         Invalid_Binary_Operator,
    //   Keyword_If
         Invalid_Binary_Operator,
    //   Keyword_Else
         Invalid_Binary_Operator,
    //   Keyword_Type
         Invalid_Binary_Operator,
    // Non-operator symbols
    //  Open_Parentheses:
         Invalid_Binary_Operator,
    //  Closed_Parentheses:
         Invalid_Binary_Operator,
    //  End_Statement:
         Invalid_Binary_Operator,
    //  Type_Indicator:
         Invalid_Binary_Operator,
    //  Comma:
         Invalid_Binary_Operator,
    // Operators
    //  Arithmetic
    //   Op_Plus:
          14,
    //   Op_Minus:
          14,
    //   Op_Div:
          15,
    //   Op_Mult:
          15,
    //   Op_Mod:
          15,
    //   Op_Rem:
          15,
    //  Logical
    //   Op_And:
          7,
    //   Op_Or:
          6,
    //   Op_Not:
          Invalid_Binary_Operator,
    //  Comparison
    //   Op_Eq:
          11,
    //   Op_Ne:
          11,
    //   Op_Lt:
          12,
    //   Op_Gt:
          12,
    //   Op_Le:
          12,
    //   Op_Ge:
          12,
    //  Bitwise
    //   Op_Left_Shift:
          13,
    //   Op_Right_Shift:
          13,
    //   Op_Bit_And:
          10,
    //   Op_Bit_Or:
          8,
    //   Op_Bit_Xor:
          9,
    //   Op_Bit_Not:
          Invalid_Binary_Operator,
    //  Range
    //   Op_Thru:
          13,
    //   Op_Upto:
          13,
    // Pseudo-Operators
    //  Op_Assign:
         Invalid_Binary_Operator,
    // Operands
    //  String_Literal:
         Operand,
    //  Char_Literal:
         Operand,
    //  Int_Literal:
         Operand,
    //  Float_Literal:
         Operand,
    //  Name:
         Operand
};

static_assert(sizeof(operator_precedence_table) / sizeof(operator_precedence_table[0])
              == int(TokenType::Name) + 1, "Table out-of-sync with TokenType enum");

constexpr Precedence precedence_of(const TokenType index)
{
    return operator_precedence_table[short(index)];
}

constexpr bool is_binary_operator(const Precedence p)
{
    return p >= 0 && p != Operand;
}


void print_error(unsigned int line_num, std::string_view message)
{
    std::cerr << "ERROR: Line " << line_num << ": "
              << message << "\n";
}

void print_error(std::string_view message)
{
    std::cerr << "ERROR: " << message << "\n";
}

void print_error_expected(std::string_view expected, Token actual)
{
    std::cerr << "ERROR: Line " << actual.line_num << ": "
              << "Expected " << expected << ", but instead found token:\n";
    std::cerr << "  " << actual;
}

void print_error_expected(unsigned int line_num, std::string_view expected,
                          const Expression* actual)
{
    std::cerr << "ERROR: Line " << line_num << ": "
              << "Expected " << expected << ", but instead found expression:\n";
    actual->print(std::cerr);
    std::cerr << '\n';
}

void assert_token_is(TokenType type, std::string_view description, Token token)
{
    if(token.type != type) {
        print_error_expected(description, token);
        exit(1);
    }
}

Expression* fold_constants(Token op, Expression* right)
{
    if(right->kind() != ExpressionKind::IntLiteral) {
        return new UnaryExpression(op.type, right);
    } else {
        auto* r_int = static_cast<IntLiteral*>(right);
        switch(op.type) {
        case TokenType::Op_Minus:
            r_int->value.negate();
            break;
        case TokenType::Op_Bit_Not:
            r_int->value.ones_complement();
            break;
        default:
            print_error_expected("unary operator that works on integer literals", op);
            exit(1);
        }
        return r_int;
    }
}

Expression* fold_constants(Expression* left, const Token& op, Expression* right)
{
    if(left->kind() != ExpressionKind::IntLiteral
       || right->kind() != ExpressionKind::IntLiteral) {
        // Can't do any folding, need TWO constant operands
        return new BinaryExpression(left, op.type, right);
    } else {
        // Fold into a single literal (uses arbitrary-precision arithmetic)
        auto* l_int = static_cast<IntLiteral*>(left);
        auto* r_int = static_cast<IntLiteral*>(right);
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
            return new BinaryExpression(left, op.type, right);
        default:
            print_error_expected("binary operator that works on integer literals", op);
            exit(1);
        }
        delete right;
        return left;
    }
}

Parser::Parser(TokenIterator input_begin, TokenIterator input_end)
    : m_input_begin(input_begin), m_input_end(input_end),
      m_range_types(sizeof(RangeType) * 64), m_lvalues(sizeof(LValue) * 64),
      m_functions(sizeof(Function) * 16), m_statements(sizeof(Statement) * 64),
      m_names_table(m_range_types, m_lvalues, m_functions)
{}

// Pratt parser
Expression* Parser::parse_expression(TokenType right_token)
{
    Expression* left_side = in_expression();
    const Precedence right_precedence = precedence_of(right_token);
    Precedence curr_precedence = precedence_of(token->type);
    while(right_precedence < curr_precedence && token->type != TokenType::End_Statement) {
        auto op = token;
        if(!is_binary_operator(curr_precedence)) {
            print_error_expected("binary operator", *token);
            exit(1);
        }
        ++token;
        Expression* right_side = parse_expression(op->type);
        left_side = fold_constants(left_side, *op, right_side);
        curr_precedence = precedence_of(token->type);
    }
    return left_side;
}

Expression* Parser::in_literal()
{
    const TokenIterator current = token++;

    switch(current->type) {
    case TokenType::String_Literal:
        return new StringLiteral(current->text);
    case TokenType::Char_Literal:
        return new CharLiteral(current->text[0]);
    case TokenType::Int_Literal:
        return new IntLiteral(current->text);
    case TokenType::Float_Literal:
        return new FloatLiteral(std::stod(current->text));
    default:
        print_error_expected("literal", *current);
        exit(1);
    }
}

Expression* Parser::in_lvalue_expression()
{
    const TokenIterator current = token++;
    assert_token_is(TokenType::Name, "variable/constant name", *current);

    // Need to find the right LValue that this LValueExpression is a usage of
    const auto match = m_names_table.find(current->text);
    if(!match) {
        print_error(current->line_num,
                    "Unknown variable/constant `" + current->text + "`");
        exit(1);
    } else if(match.value().name_type != NameType::LValue) {
        print_error("Expected name of variable/constant, but `"
                    + current->text + "` is already being used as a name");
        exit(1);
    } else {
        auto *new_lvalue_expr = new LValueExpression(current->text, match.value().lvalue);
        new_lvalue_expr->name = current->text;
        return new_lvalue_expr;
    }
}

Expression* Parser::in_expression()
{
    switch(token->type) {
    case TokenType::Name:
        if(std::next(token)->type == TokenType::Open_Parentheses) {
            return in_function_call();
        } else {
            return in_lvalue_expression();
        }
    case TokenType::Open_Parentheses:
        return in_parentheses();
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

Expression* Parser::in_function_call()
{
    assert_token_is(TokenType::Name, "function name", *token);

    // First, assign the function call's name
    auto* new_function_call = new FunctionCall(token->text);
    const auto match = m_names_table.find(token->text);
    if(!match) {
        // If the function hasn't been declared yet, add it provisionally to name table
        // to be filled in (hopefully) later
        new_function_call->function = m_names_table.add_function(token->text);
        m_names_table.add_unresolved(new_function_call);
    } else if(match.value().name_type == NameType::DeclaredFunct) {
        // A temp declaration (one without definition) has been declared, but haven't
        // resolved its definition yet
        new_function_call->function = match.value().function;
        m_names_table.add_unresolved(new_function_call);
    } else if(match.value().name_type == NameType::Funct) {
        // Found a function with a full definition
        new_function_call->function = match.value().function;
    } else {
        print_error(token->line_num, "Expected `" + token->text
                    + "` to be a function name, but it is defined as another kind of name");
        exit(1);
    }

    ++token;
    assert_token_is(TokenType::Open_Parentheses,
                    "'(' token before function call argument list", *token);

    // Next, parse the arguments, each of which should be some sort of expression
    ++token;
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
            new_function_call->arguments.emplace_back(parse_expression(TokenType::Comma));
        }
    }
    print_error(token->line_num, "Function call definition ended early");
    exit(1);
}

Expression* Parser::in_parentheses()
{
    assert_token_is(TokenType::Open_Parentheses,
                    "opening parentheses for an expression group", *token);

    ++token;
    Expression* result = parse_expression(TokenType::Closed_Parentheses);
    if(token->type != TokenType::Closed_Parentheses) {
        print_error_expected("closing parentheses for an expression group", *token);
        exit(1);
    }

    ++token;
    return result;
}

LValue* Parser::in_lvalue_declaration()
{
    assert_token_is(TokenType::Name, "the name of an lvalue", *token);
    if(auto name_exists = m_names_table.find(token->text); name_exists) {
        print_error(token->line_num, "`" + token->text
                    + "` cannot be used as an lvalue name. It is already defined as a name");
        exit(1);
    }
    LValue* new_lvalue = m_names_table.add_lvalue(token->text);

    ++token;
    assert_token_is(TokenType::Type_Indicator, "`:` before typename", *token);

    ++token;
    if(token->type == TokenType::Keyword_Const) {
        // `constant` keyword marks kind of lvalue
        new_lvalue->is_mutable = false;
        ++token;
    }
    assert_token_is(TokenType::Name, "typename", *token);

    const auto match = m_names_table.find(token->text);
    if(!match) {
        // If the type hasn't been declared yet, add it provisionally to name table
        // to be filled in (hopefully) later
        new_lvalue->type = m_names_table.add_type(token->text);
        m_names_table.add_unresolved(new_lvalue);
    } else if(match.value().name_type == NameType::DeclaredType) {
        // A temp type (one that lacks a definition) has already been declared, but
        // the actual definition of it hasn't been resolved yet
        new_lvalue->type = match.value().range_type;
        m_names_table.add_unresolved(new_lvalue);
    } else if(match.value().name_type == NameType::Type) {
        new_lvalue->type = match.value().range_type;
    } else {
        print_error(token->line_num, "Expected `" + token->text
                    + "` to be a typename, but it is defined as another kind of name");
        exit(1);
    }

    return new_lvalue;
}

Initialization* Parser::in_initialization()
{
    assert_token_is(TokenType::Keyword_Let, "keyword `let`", *token);
    unsigned int line_num = token->line_num;

    // Add this new lvalue to list of tracked names
    ++token;
    auto *new_statement =
        m_statements.make<Initialization>(line_num, in_lvalue_declaration());

    ++token;
    if(token->type == TokenType::End_Statement) {
        // Declaration is valid, just no initial value was set
        ++token;
        return new_statement;
    } else if(token->type != TokenType::Op_Assign) {
        print_error_expected("assignment operator or ';'", *token);
        exit(1);
    } else {
        ++token;
        // Set the expression after the assignment operator to be the subexpression
        // in the statement
        new_statement->expression = Magnum::pointer<Expression>(parse_expression());
        ++token;
        return new_statement;
    }
}

Assignment* Parser::in_assignment()
{
    auto lval_match = m_names_table.find(token->text);
    if(!lval_match) {
        print_error(token->line_num, "`" + token->text
                    + "` is not a variable name and so cannot be assigned to");
        exit(1);
    } else if(lval_match->name_type != NameType::LValue) {
        print_error(token->line_num, "Expected `" + token->text
                    + "` to be a variable, but it is defined as another kind of name");
        exit(1);
    } else if(!lval_match->lvalue->is_mutable) {
        print_error(token->line_num, "cannot assign to constant `" + token->text + "`");
        exit(1);
    }

    unsigned int line_num = token->line_num;
    std::advance(token, 2); // skip varname and `=` operator
    Expression* expr = parse_expression();
    ++token;
    return m_statements.make<Assignment>(line_num, expr, lval_match->lvalue);
}

Statement* Parser::in_statement()
{
    switch(token->type) {
    case TokenType::Keyword_If:
        return in_if_block();
    case TokenType::Keyword_Let:
        return in_initialization();
    default:
        if(std::next(token)->type == TokenType::Op_Assign) {
            return in_assignment();
        } else {
            return in_basic_statement();
        }
    };
}

BasicStatement* Parser::in_basic_statement()
{
    auto *new_statement =
        m_statements.make<BasicStatement>(token->line_num, parse_expression());

    assert_token_is(TokenType::End_Statement, "end of statement (a.k.a. `;`)", *token);
    ++token;
    return new_statement;
}

IfBlock* Parser::in_if_block()
{
    assert_token_is(TokenType::Keyword_If, "keyword if", *token);
    ++token;

    m_names_table.open_scope();
    unsigned int line_num = token->line_num;
    // First, parse the if-condition
    auto *new_block = m_statements.make<IfBlock>(line_num, parse_expression());
    assert_token_is(TokenType::Keyword_Do,
                    "keyword `do` following `if` condition", *token);

    // Next, parse the statements inside the if-block
    ++token;
    while(token != m_input_end) {
        if(token->type != TokenType::Keyword_End) {
            // Found a statement, parse it
            new_block->statements.push_back(in_statement());
        } else {
            // End of block
            m_names_table.close_scope();
            if(std::next(token)->type == TokenType::Keyword_If) {
                // Optional keyword to show end of if-block
                ++token;
            }
            ++token;
            assert_token_is(TokenType::End_Statement,
                            "closing `;` or an end label after end of if-block",
                            *token);
            ++token;
            break;
        }
    }

    return new_block;
}

void Parser::in_function_definition()
{
    ++token;
    assert_token_is(TokenType::Name, "name to follow `function keyword`", *token);

    // First, set the function's name, making sure it isn't being used for
    // anything else
    const auto match = m_names_table.find(token->text);
    if(match && match.value().name_type != NameType::DeclaredFunct) {
        print_error(token->line_num, "Name `" + token->text + "` is"
                    " already in use");
        exit(1);
    }
    Function new_funct{token->text};

    ++token;
    assert_token_is(TokenType::Open_Parentheses,
                    "`(` to follow name of function in definition", *token);

    m_names_table.open_scope();
    // Next, parse the parameters
    ++token;
    while(token != m_input_end) {
        if(token->type == TokenType::Name) {
            // Add a new parameter declaration
            new_funct.parameters.push_back(in_lvalue_declaration());

            ++token;
            // Check for a comma separator after typename
            if(token->type == TokenType::Closed_Parentheses) {
                // Put back so next iteration can handle end of parameter list
                --token;
            } else if(token->type != TokenType::Comma) {
                print_error_expected("comma to follow the parameter", *token);
                exit(1);
            }
        } else if(token->type == TokenType::Closed_Parentheses) {
            // Next, look for `is` keyword
            ++token;
            assert_token_is(TokenType::Keyword_Is,
                            "keyword `is` to follow parameters of the function", *token);
            break;
        } else {
            print_error_expected("parameter name", *token);
            exit(1);
        }

        ++token;
    }

    // Finally, parse the body of the function
    ++token;
    while(token != m_input_end) {
        if(token->type != TokenType::Keyword_End) {
            // Found a statement, parse it
            new_funct.statements.push_back(in_statement());
        } else {
            // End of function
            m_names_table.close_scope();
            if(std::next(token)->type == TokenType::Name
               && std::next(token)->text == new_funct.name) {
                // Optional end label for function; comes after `end`
                ++token;
            }
            ++token;
            assert_token_is(TokenType::End_Statement,
                            "end of statement (a.k.a. `;`) or an end label", *token);
            Function* ptr = m_names_table.add_function(std::move(new_funct));
            m_function_list.push_back(ptr);
            ++token;
            return;
        }
    }
}

void Parser::in_range_type_definition(const std::string& type_name)
{
    Magnum::Pointer<Expression> expr = Magnum::pointer(parse_expression());
    if(expr->kind() != ExpressionKind::Binary) {
        print_error_expected("binary expression with operator `thru` or `upto`", *token);
        exit(1);
    }
    auto* range_expr = static_cast<const BinaryExpression*>(expr.get());
    if(range_expr->op != TokenType::Op_Thru && range_expr->op != TokenType::Op_Upto) {
        print_error_expected("binary expression with operator `thru` or `upto`", *token);
        exit(1);
    }

    if(range_expr->left->kind() != ExpressionKind::IntLiteral) {
        print_error_expected(token->line_num,
                             "integer constant expression as the range's lower bound",
                             range_expr->left.get());
        exit(1);
    } else if(range_expr->right->kind() != ExpressionKind::IntLiteral) {
        print_error_expected(token->line_num,
                             "integer constant expression as the range's upper bound",
                             range_expr->right.get());
        exit(1);
    }
    auto* left_expr = static_cast<const IntLiteral*>(range_expr->left.get());
    auto* right_expr = static_cast<const IntLiteral*>(range_expr->right.get());

    // TODO: Support arbitrary expressions made of arithmetic
    //  operators/parentheses/negations/bitwise operators
    multi_int lower_limit{left_expr->value};
    multi_int upper_limit{right_expr->value};

    if(upper_limit < lower_limit) {
        print_error(token->line_num, "Upper limit of range is lower than the lower limit");
        exit(1);
    } else if(range_expr->op == TokenType::Op_Upto && upper_limit == lower_limit) {
        print_error(token->line_num, "In `upto` range, lower limit cannot be same as upper limit");
        exit(1);
    }

    RangeType* ptr = m_names_table.add_type(RangeType{{type_name},
                                                      Range{lower_limit, upper_limit}});
    m_range_type_list.push_back(ptr);
}

void Parser::in_type_definition()
{
    assert_token_is(TokenType::Keyword_Type, "keyword `type`", *token);
    ++token;
    assert_token_is(TokenType::Name, "typename", *token);

    if(const auto match = m_names_table.find(token->text);
       match && match.value().name_type != NameType::DeclaredType) {
        print_error(token->line_num, "Name: " + token->text + " already in use");
        exit(1);
    }

    const auto& type_name = token->text;
    ++token;
    assert_token_is(TokenType::Keyword_Is, "keyword `is`", *token);

    // TODO: add ability to distinguish between new distinct types and new subtypes
    ++token;
    if(token->type == TokenType::Keyword_Range) {
        // Handle discrete types
        ++token; // Eat keyword `range`
        in_range_type_definition(type_name);
    } else {
        // TODO: handle arrays/record types here
        print_error_expected("keyword range", *token);
        exit(1);
    }

    assert_token_is(TokenType::End_Statement, "end of statement (a.k.a. `;`)", *token);
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
        default:
            print_error(token->line_num, "Unexpected token:");
            std::cerr << *token << '\n';
            exit(1);
        }
    }

    // Check that there are no unresolved types/functions
    m_names_table.validate_names();
}

std::ostream& operator<<(std::ostream& output, const Parser& parser)
{
    for(const auto *function_definition : parser.m_function_list) {
         output << *function_definition << '\n';
    }
    return output;
}


SymbolTable::SymbolTable(MemoryPool& range_types, MemoryPool& lvalues,
                         MemoryPool& functions)
    : m_range_types(range_types), m_lvalues(lvalues), m_functions(functions)
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


std::optional<SymbolInfo>
SymbolTable::search_for_definition(const std::string& name, NameType kind) const
{
    int scope_index = m_curr_scope;
    while(scope_index >= 0) {
        const Scope& scope = m_scopes[scope_index];
        const auto match = scope.symbols.find(name);
        if(match != scope.symbols.end()
           && match->second.name_type == kind) {
            return {match->second};
        } else {
            scope_index = scope.parent_index;
        }
    }
    return {};
}

LValue* SymbolTable::add_lvalue(const std::string& name)
{
    LValue* ptr = m_lvalues.make<LValue>(name);
    m_scopes[m_curr_scope].symbols[name] = SymbolInfo{NameType::LValue, ptr};
    return ptr;
}

RangeType* SymbolTable::add_type(RangeType&& type)
{
    RangeType* ptr = m_range_types.make<RangeType>(type);
    m_scopes[m_curr_scope].symbols[type.name]
        = SymbolInfo{NameType::Type, ptr};
    return ptr;
}


RangeType* SymbolTable::add_type(const std::string& name)
{
    RangeType* ptr = m_range_types.make<RangeType>(name);
    m_scopes[m_curr_scope].symbols[name] = SymbolInfo{NameType::DeclaredType, ptr};
    // Return temp type until the type of this lvalue gets resolved
    return ptr;
}

Function* SymbolTable::add_function(Function&& function)
{
    Function* ptr = m_functions.make<Function>(function);
    m_scopes[m_curr_scope].symbols[function.name] = SymbolInfo{NameType::Funct, ptr};
    return ptr;
}

Function* SymbolTable::add_function(const std::string& name)
{
    Function* ptr = m_functions.make<Function>(name);
    m_scopes[m_curr_scope].symbols[name] = SymbolInfo{NameType::DeclaredFunct, ptr};
    return ptr;
}

void SymbolTable::add_unresolved(LValue* lvalue)
{
    m_scopes[m_curr_scope].lvalues_type_unresolved.push_back(lvalue);
}

void SymbolTable::add_unresolved(FunctionCall* funct)
{
    m_scopes[m_curr_scope].unresolved_funct_calls.push_back(funct);
}

void SymbolTable::validate_names()
{
    // Check that all functions and types in this module have a definition.
    // If they don't, try to find one/fix-up the symbol table

    // TODO: Have mechanism where types/functions in other modules are resolved
    int old_curr_scope = m_curr_scope;
    for(int scope_index = 0; scope_index < m_scopes.size(); ++scope_index) {
        m_curr_scope = scope_index;
        Scope& scope = m_scopes[scope_index];
        // Try and resolve the types of lvalues whose types were not declared beforehand
        for(LValue* lvalue : scope.lvalues_type_unresolved) {
            std::optional<SymbolInfo> match
                = search_for_definition(lvalue->type->name, NameType::Type);
            if(!match) {
                print_error("Type `" + lvalue->type->name
                            + "` is used but has no definition");
                exit(1);
            } else {
                // Update to the newly-defined type (replacing the temp type)
                lvalue->type = match->range_type;
            }
        }
        scope.lvalues_type_unresolved.clear();

        // Try and resolve function calls to functions declared later
        for(FunctionCall* funct_call : scope.unresolved_funct_calls) {
            std::optional<SymbolInfo> match
                = search_for_definition(funct_call->name, NameType::Funct);
            if(!match) {
                print_error("Function `" + funct_call->name
                            + "` is used but has no definition");
                exit(1);
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
