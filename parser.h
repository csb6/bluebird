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
#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include "token.h"
#include "ast.h"
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include "memory-pool.h"
#include <iosfwd>

struct SymbolInfo {
    NameType name_type;
    union {
        RangeType* range_type;
        LValue* lvalue = nullptr;
        Function* function;
    };

    SymbolInfo() {}
    explicit SymbolInfo(NameType name) : name_type(name) {}
    SymbolInfo(NameType name, RangeType* type)
        : name_type(name), range_type(type)
    {}
    SymbolInfo(NameType name, LValue* lv)
        : name_type(name), lvalue(lv)
    {}
    SymbolInfo(NameType name, Function* f)
        : name_type(name), function(f)
    {}
};

struct Scope {
    using symbol_iterator = std::unordered_map<std::string, SymbolInfo>::iterator;
    int parent_index;
    std::unordered_map<std::string, SymbolInfo> symbols{};
    std::vector<LValue*> lvalues_type_unresolved{};
    std::vector<FunctionCall*> unresolved_funct_calls{};
};

class SymbolTable {
public:
    SymbolTable(MemoryPool& range_types, MemoryPool& lvalues,
                MemoryPool& functions);
    void open_scope();
    void close_scope();
    std::optional<SymbolInfo> find(const std::string& name) const;

    // All add functions assume the name isn't already used for something else
    void       add_lvalue(LValue*);
    RangeType* add_type(const std::string& name,
                        const multi_int& lower_limit, const multi_int& upper_limit);
    // Add a temporary type that lacks a definition
    RangeType* add_type(const std::string& name);
    Function*  add_function(Function&&);
    Function*  add_function(const std::string& name);

    void add_unresolved(LValue*);
    void add_unresolved(FunctionCall*);

    // Checking that no names are declared but not defined (or imported)
    void validate_names();
private:
    // Scope tree
    std::vector<Scope> m_scopes;
    int m_curr_scope;

    // AST entities
    MemoryPool &m_range_types;
    MemoryPool &m_lvalues;
    MemoryPool &m_functions;

    std::optional<SymbolInfo>
    search_for_definition(const std::string& name, NameType) const;
    std::optional<SymbolInfo>
    search_for_funct_definition(const std::string& name) const;
};

class Parser {
public:
    using TokenIterator = std::vector<Token>::const_iterator;
private:
    TokenIterator m_input_begin;
    TokenIterator m_input_end;
    TokenIterator token;

    MemoryPool m_range_types;
    MemoryPool m_lvalues;
    MemoryPool m_functions;
    MemoryPool m_statements;
    SymbolTable m_names_table;

    // Points to memory in m_functions
    std::vector<Function*> m_function_list;
    // Points to memory in m_range_types
    std::vector<RangeType*> m_range_type_list;

    Expression* parse_expression(TokenType right_token = TokenType::Keyword_Is);
    // Helpers
    LValue* in_lvalue_declaration();
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
    // Handle each function definition
    void in_function_definition();
    // Types
    void in_range_type_definition(const std::string& type_name);
    void in_type_definition();
public:
    Parser(TokenIterator input_begin,
           TokenIterator input_end);
    void run();
    const auto& functions() const { return m_function_list; }
    const auto& types() const { return m_range_type_list; }
    const auto& names_table() const { return m_names_table; }

    friend std::ostream& operator<<(std::ostream&, const Parser&);
};
#endif
