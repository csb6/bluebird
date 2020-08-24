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
    std::cerr << "Line " << line_num << ": "
              << message << "\n";
}

void print_error(std::string_view message)
{
    std::cerr << message << "\n";
}

void print_error_expected(std::string_view expected, Token actual)
{
    std::cerr << "Line " << actual.line_num << ": "
              << "Expected " << expected << ", but instead found token:\n";
    std::cerr << actual;
}

void print_error_expected(unsigned int line_num, std::string_view expected,
                          const Expression* actual)
{
    std::cerr << "Line " << line_num << ": "
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

Parser::Parser(TokenIterator input_begin,
               TokenIterator input_end)
    : m_input_begin(input_begin), m_input_end(input_end)
{
    // Built-in functions/names (eventually, the standard library)
    m_names_table.add("print", NameType::Funct);
    m_names_table.add("max", NameType::Funct);
}

// Pratt parser
Expression* Parser::parse_expression(TokenType right_token)
{
    Expression* left_side = in_expression();
    const Precedence right_precedence = precedence_of(right_token);
    Precedence curr_precedence = precedence_of(token->type);
    while(right_precedence < curr_precedence && token->type != TokenType::End_Statement) {
        TokenType op = token->type;
        if(!is_binary_operator(curr_precedence)) {
            print_error_expected("binary operator", *token);
            exit(1);
        }
        ++token;
        Expression* right_side = parse_expression(op);
        left_side = new BinaryExpression(left_side, op, right_side);
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
        return new IntLiteral(std::stoi(current->text));
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

    const auto match = m_names_table.find(current->text);
    if(!match) {
        print_error(current->line_num, "Unknown lvalue `" + current->text + "`");
        exit(1);
    } else if(match.value() != NameType::LValue) {
        print_error("Expected name of variable/constant, but `"
                    + current->text + "` is already being used as a name");
        exit(1);
    }

    auto new_lvalue_expr = new LValueExpression();
    new_lvalue_expr->name = current->text;
    return new_lvalue_expr;
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
        TokenType op = token->type;
        ++token;
        Expression* right = in_expression();
        return new UnaryExpression(op, right);
    }
    default:
        return in_literal();
    }
}

FunctionCall* Parser::in_function_call()
{
    auto* new_function_call = new FunctionCall();

    assert_token_is(TokenType::Name, "function name", *token);

    // First, assign the function call's name
    const auto match = m_names_table.find(token->text);
    if(!match) {
        // If the function hasn't been declared yet, add it provisionally to name table
        // to be filled in (hopefully) later
        m_names_table.add(token->text, NameType::DeclaredFunct);
    } else if(match.value() != NameType::Funct && match.value() != NameType::DeclaredFunct) {
        print_error(token->line_num, "Expected `" + token->text
                    + "` to be a function name, but it is defined as another kind of name");
        exit(1);
    }

    new_function_call->name = token->text;

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

Magnum::Pointer<LValue> Parser::in_lvalue_declaration()
{
    assert_token_is(TokenType::Name, "the name of an lvalue", *token);
    Magnum::Pointer<LValue> new_lvalue;
    new_lvalue = Magnum::pointer<LValue>();
    if(auto name_exists = m_names_table.find(token->text); name_exists) {
        print_error(token->line_num, "`" + token->text + "` cannot be used as an lvalue name. It is already defined as another kind of name");
        exit(1);
    }
    new_lvalue->name = token->text;

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
        m_names_table.add(token->text, NameType::DeclaredType);
    } else if(match.value() != NameType::Type && match.value() != NameType::DeclaredType) {
        print_error(token->line_num, "Expected `" + token->text
                    + "` to be a typename, but it is defined as another kind of name");
        exit(1);
    }

    new_lvalue->type = token->text;
    return new_lvalue;
}

Magnum::Pointer<Initialization> Parser::in_initialization()
{
    assert_token_is(TokenType::Keyword_Let, "keyword `let`", *token);

    auto new_statement = Magnum::pointer<Initialization>(token->line_num);

    // Add this new lvalue to list of tracked names
    {
        ++token;
        Magnum::Pointer<LValue> new_lvalue = in_lvalue_declaration();
        if(const auto match = m_names_table.find(new_lvalue->name); match) {
            print_error(token->line_num, "Name `" + new_lvalue->name + "` is already in use");
            exit(1);
        }
        m_names_table.add(new_lvalue->name, NameType::LValue);
        m_lvalues.push_back(std::move(new_lvalue));
        new_statement->target = m_lvalues.back().get();
    }

    ++token;
    if(token->type == TokenType::End_Statement) {
        // Declaration is valid, just no initial value was set
        ++token;
        return new_statement;
    } else if(token->type != TokenType::Op_Assign) {
        print_error_expected("assignment operator or ';'", *token);
        exit(1);
    }

    ++token;
    // Set the expression after the assignment operator to be the subexpression
    // in the statement
    new_statement->expression = Magnum::pointer<Expression>(parse_expression());

    ++token;
    return new_statement;
}

Magnum::Pointer<Statement> Parser::in_statement()
{
    switch(token->type) {
    case TokenType::Keyword_If:
        return in_if_block();
    case TokenType::Keyword_Let:
        return in_initialization();
    default:
        return in_basic_statement();
    };
}

Magnum::Pointer<BasicStatement> Parser::in_basic_statement()
{
    auto new_statement = Magnum::pointer<BasicStatement>(token->line_num);
    new_statement->expression = Magnum::pointer<Expression>(parse_expression());

    assert_token_is(TokenType::End_Statement, "end of statement (a.k.a. `;`)", *token);
    ++token;
    return new_statement;
}

Magnum::Pointer<IfBlock> Parser::in_if_block()
{
    assert_token_is(TokenType::Keyword_If, "keyword if", *token);

    ++token;
    assert_token_is(TokenType::Open_Parentheses, "`(`", *token);

    ++token;
    auto new_block = Magnum::pointer<IfBlock>(token->line_num);

    m_names_table.open_scope();

    // First, parse the if-condition
    new_block->condition = Magnum::pointer<Expression>(
        parse_expression(TokenType::Closed_Parentheses));

    assert_token_is(TokenType::Closed_Parentheses,
                    "`)` to close `if` condition", *token);

    ++token;
    assert_token_is(TokenType::Keyword_Do, "keyword `do`", *token);

    // Next, parse the statements inside the if-block
    ++token;
    while(token != m_input_end) {
        if(token->type != TokenType::Keyword_End) {
            // Found a statement, parse it
            new_block->statements.push_back(in_statement());
        } else {
            // End of block
            m_names_table.close_scope();
            ++token; // Take the `if` associated with `end`
            if(token->type != TokenType::Keyword_If) {
                print_error(token->line_num,
                            "No closing `end if` for if-block");
                exit(1);
            }
            ++token;
            break;
        }
    }

    return new_block;
}

void Parser::in_function_definition()
{
    Function new_funct;

    // First, set the function's name, making sure it isn't being used for
    // anything else
    ++token;
    assert_token_is(TokenType::Name, "name to follow `function keyword`", *token);

    const auto match = m_names_table.find(token->text);
    if(!match) {
        new_funct.name = token->text;
        m_names_table.add(new_funct.name, NameType::Funct);
    } else if(match.value() == NameType::DeclaredFunct) {
        new_funct.name = token->text;
        m_names_table.update(new_funct.name, NameType::Funct);
    } else {
        print_error(token->line_num, "Name `" + token->text + "` is"
                    + " already in use");
        exit(1);
    }

    ++token;
    assert_token_is(TokenType::Open_Parentheses,
                    "`(` to follow name of function in definition", *token);

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

    m_names_table.open_scope();

    // Finally, parse the body of the function
    ++token;
    while(token != m_input_end) {
        if(token->type != TokenType::Keyword_End) {
            // Found a statement, parse it
            new_funct.statements.push_back(in_statement());
        } else {
            // End of function
            m_names_table.close_scope();
            ++token; // Take the name associated with `end` (e.g. `main` in `end main`)
            if(token->type != TokenType::Name && token->text != new_funct.name) {
                print_error(token->line_num,
                            "No matching `end " + new_funct.name + "` for"
                            + " `function` " + new_funct.name);
                exit(1);
            }

            ++token;
            assert_token_is(TokenType::End_Statement,
                            "end of statement (a.k.a. `;`)", *token);
            m_functions.push_back(std::move(new_funct));
            ++token;
            return;
        }
    }
}

void Parser::in_type_definition()
{
    assert_token_is(TokenType::Keyword_Type, "keyword `type`", *token);
    ++token;
    assert_token_is(TokenType::Name, "typename", *token);

    if(const auto match = m_names_table.find(token->text);
       match && match.value() != NameType::DeclaredType) {
        print_error(token->line_num, "Name: " + token->text + " already in use");
        exit(1);
    }

    m_names_table.add(token->text, NameType::Type);
    m_types.push_back({token->text});

    ++token;
    assert_token_is(TokenType::Keyword_Is, "keyword `is`", *token);

    // TODO: add ability to distinguish between new distinct types and new subtypes
    ++token;
    if(token->type == TokenType::Keyword_Range) {
        // Handle discrete types
        ++token;
        Magnum::Pointer<Expression> expr = Magnum::pointer(parse_expression());
        if(expr->expr_type() != ExpressionType::Binary) {
            print_error_expected("binary expression with operator `thru` or `upto`", *token);
            exit(1);
        }
        const auto* range_expr = static_cast<const BinaryExpression*>(expr.get());
        if(range_expr->op != TokenType::Op_Thru && range_expr->op != TokenType::Op_Upto) {
            print_error_expected("binary expression with operator `thru` or `upto`", *token);
            exit(1);
        }

        // TODO: compile-time eval both sides of the range operator. For now, only allow
        //  int literals to keep things simple. Also need to add support for `-` operator
        //  for negative integer literals
        if(range_expr->left->expr_type() != ExpressionType::IntLiteral) {
            print_error_expected(token->line_num, "integer literal", range_expr->left.get());
            exit(1);
        } else if(range_expr->right->expr_type() != ExpressionType::IntLiteral) {
            print_error_expected(token->line_num, "integer literal", range_expr->right.get());
            exit(1);
        }

        // TODO: add the range to SymbolTable::m_discrete types, then update the scope
        // table so that the current scope points to the type's definition/its unique
        // id
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
            // Found start of a function definition
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
    for(const auto& function_definition : parser.m_functions) {
        output << function_definition << '\n';
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
    ScopeTable new_scope = {m_curr_scope};
    m_scopes.push_back(new_scope);
    m_curr_scope = m_scopes.size() - 1;
}

void SymbolTable::close_scope()
{
    m_curr_scope = m_scopes[m_curr_scope].parent;
}


std::optional<NameType> SymbolTable::find_id(short name_id) const
{
    short scope_index = m_curr_scope;
    while(scope_index != -1) {
        const ScopeTable& scope = m_scopes[scope_index];
        auto match = scope.symbols.find(name_id);
        if(match != scope.symbols.end()) {
            // Found match in this scope
            return {match->second};
        } else {
            // No match, try searching the containing scope
            scope_index = scope.parent;
        }
    }
    return {};
}

std::pair<std::string, short> SymbolTable::find_name(short name_id) const
{
    // Map name id back to its string name 
    return *std::find_if(m_ids.begin(), m_ids.end(),
                         [name_id](const auto& each) {
                             return each.second == name_id;
                         });
}

std::optional<NameType> SymbolTable::find(const std::string& name) const
{
    auto it = m_ids.find(name);
    if(it == m_ids.end()) {
        // Identifier not used anywhere
        return {};
    }

    return find_id(it->second);
}

bool SymbolTable::add(const std::string& name, NameType type)
{
    auto it = m_ids.find(name);
    short name_id;
    if(it == m_ids.end()) {
        name_id = m_curr_symbol_id++;
        m_ids[name] = name_id;
    } else {
        name_id = it->second;
    }

    auto match = find_id(name_id);
    if(!match) {
        // No conflicts here; add the symbol
        m_scopes[m_curr_scope].symbols[name_id] = type;
        return true;
    } else {
        // Symbol already used as a name somewhere else in the
        // visible scopes
        return false;
    }
}

void SymbolTable::update(const std::string& name, NameType type)
{
    short name_id = m_ids[name];
    m_scopes[m_curr_scope].symbols[name_id] = type;
}

void delete_id(ScopeTable& scope, short name_id)
{
    scope.symbols.erase(name_id);
}

void SymbolTable::validate_names()
{
    // Check that all functions and types in this module have a definition.
    // If they don't, try to find one/fix-up the symbol table

    // TODO: Have mechanism where types/functions in other modules are resolved
    for(auto& scope : m_scopes) {
        for(const auto[name_id, name_type] : scope.symbols) {
            switch(name_type) {
            case NameType::DeclaredFunct: {
                // First, try to resolve the declared function to a definition
                auto body = find_id(name_id);
                if(!body) {
                    const auto name = find_name(name_id);
                    std::cerr << "Error: Function `" << name.first
                              << "` is declared but has no body\n";
                    exit(1);
                } else {
                    // If a definition was found, then don't need this temporary
                    // 'DeclaredFunct' declaration anymore
                    delete_id(scope, name_id);
                }
                break;
            }
            case NameType::DeclaredType: {
                // First, try to resolve the declared type to a definition
                auto definition = find_id(name_id);
                if(!definition) {
                    const auto name = find_name(name_id);
                    std::cerr << "Error: Type `" << name.first
                              << "` is declared but has no definition\n";
                    exit(1);
                } else {
                    // If a definition was found, then don't need this temporary
                    // 'DeclaredType' declaration anymore
                    delete_id(scope, name_id);
                }
                break;
            }
            default:
                // All other kinds of names are acceptable
                break;
            }
        }
    }
}
