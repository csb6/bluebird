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
#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include "token.h"
#include "ast.h"
#include <optional>
#include <unordered_map>

struct SymbolInfo {
    NameType kind;
    union {
        Type* type;
        LValue* lvalue = nullptr;
        Function* function;
    };

    SymbolInfo() {}
    explicit SymbolInfo(NameType name) : kind(name) {}
    SymbolInfo(NameType name, Type* t) : kind(name), type(t) {}
    SymbolInfo(NameType name, LValue* lv) : kind(name), lvalue(lv) {}
    SymbolInfo(NameType name, Function* f) : kind(name), function(f) {}
};

struct Scope {
    using symbol_iterator = std::unordered_map<std::string, SymbolInfo>::iterator;
    int parent_index;
    std::unordered_map<std::string, SymbolInfo> symbols{};
    std::vector<LValue*> lvalues_type_unresolved{};
    std::vector<Function*> unresolved_return_type_functs{};
    std::vector<FunctionCall*> unresolved_funct_calls{};
};

class SymbolTable {
public:
    SymbolTable();
    void open_scope();
    void close_scope();
    std::optional<SymbolInfo> find(const std::string& name) const;

    // All add functions assume the name isn't already used for something else
    void add_lvalue(LValue*);
    void add_type(Type*);
    void add_function(Function*);

    void add_unresolved(LValue*);
    void add_unresolved(FunctionCall*);
    void add_unresolved_return_type(Function*);

    void resolve_usages();
private:
    // Scope tree
    std::vector<Scope> m_scopes;
    int m_curr_scope;

    std::optional<SymbolInfo> find(const std::string& name, NameType) const;
    std::optional<SymbolInfo>
    search_for_funct_definition(const std::string& name) const;
};

class Parser {
public:
    using TokenIterator = std::vector<Token>::const_iterator;
private:
    TokenIterator m_input_begin, m_input_end, token;
    SymbolTable m_names_table;

    std::vector<Magnum::Pointer<Function>> m_function_list;
    std::vector<Magnum::Pointer<RangeType>> m_range_type_list;
    std::vector<Magnum::Pointer<Initialization>> m_global_var_list;

    Expression* parse_expression(TokenType right_token = TokenType::Keyword_Is);
    // Helpers
    LValue* in_lvalue_declaration();
    void in_return_type(Function*);
    // Handle each type of expression
    Expression* in_literal();
    Expression* in_lvalue_expression();
    Expression* in_parentheses();
    Expression* in_expression();
    Expression* in_function_call();
    // Handle each type of statement
    Statement* in_statement();
    BasicStatement* in_basic_statement();
    Initialization* in_initialization();
    Assignment* in_assignment();
    IfBlock* in_if_block();
    Block* in_else_block();
    WhileLoop* in_while_loop();
    ReturnStatement* in_return_statement();
    // Handle each function definition
    void in_function_definition();
    // Types
    void in_range_type_definition(const std::string& type_name);
    void in_type_definition();
public:
    Parser(TokenIterator input_begin,
           TokenIterator input_end);
    void run();
    auto& functions() { return m_function_list; }
    auto& types() { return m_range_type_list; }
    auto& global_vars() { return m_global_var_list; }
    auto& names_table() { return m_names_table; }

    friend std::ostream& operator<<(std::ostream&, const Parser&);
};
#endif
