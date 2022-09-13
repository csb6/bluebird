/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2022  Cole Blakley

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

/* This file contains a class that parses a stream of tokens into an AST.
   As it builds the AST, it maintains a symbol table and handles out-of-order
   definitions of types and functions, which are then resolved once the
   entire module has been parsed. It contains a variety of syntactic checks.

   The parser uses the recursive descent strategy, additionally using a Pratt parser
   for expressions. The operator precedence table can be found in this file's
   implementation (parser.cpp).

   By the end of this stage, the AST should be nearly complete, with all function
   calls, type usages, and variable usages resolved to their definitions.
*/

/* Holds the results of a query into the SymbolTable, indicating what kind
   of value (if any) was found to be associated with a particular name */
struct SymbolInfo {
    NameType kind; // See ast.h
    union {
        Type* type;
        Variable* variable = nullptr;
        Function* function;
    };

    SymbolInfo() {}
    explicit SymbolInfo(NameType name) : kind(name) {}
    SymbolInfo(NameType name, Type* t) : kind(name), type(t) {}
    SymbolInfo(NameType name, Variable* var) : kind(name), variable(var) {}
    SymbolInfo(NameType name, Function* f) : kind(name), function(f) {}
};

/* A collection of symbols that may be a parent to other scopes. Symbols
   in this scope are only viewable from this scope and its child scopes.
   None of this scope's ancestors have symbols with the same name (i.e.
   shadowing is not allowed) */
struct Scope {
    using symbol_iterator = std::unordered_map<std::string, SymbolInfo>::iterator;
    int parent_index;
    std::unordered_map<std::string, SymbolInfo> symbols{};
    // These 'unresolved' lists are maintained for each scope until the end of
    // this module's parsing process, at which point their definitions are resolved
    std::vector<FunctionCall*> unresolved_funct_calls{};
    std::vector<Type**> unresolved_types{};
};

/* Holds a tree of scopes which can be queried and added to */
class SymbolTable {
public:
    SymbolTable();
    /* Pushes a new scope */
    void open_scope();
    /* Change current scope to be parent scope */
    void close_scope();
    std::optional<SymbolInfo> find(const std::string& name) const;

    // All add functions assume the name isn't already used for something else
    void add_var(Variable*);
    void add_type(Type*);
    void add_function(Function*);

    void add_unresolved(Variable*);
    void add_unresolved(FunctionCall*);
    void add_unresolved_return_type(Function*);

    void resolve_usages();
private:
    // Scope tree
    std::vector<Scope> m_scopes;
    // Index into m_scopes (starts at 0)
    int m_curr_scope;

    std::optional<SymbolInfo> find(const std::string& name, NameType) const;
    std::optional<SymbolInfo>
    search_for_funct_definition(const std::string& name) const;
};

/* Contains the recursive descent and Pratt parser, generates and owns the AST 
   and related data structures */
class Parser {
public:
    using TokenIterator = std::vector<Token>::const_iterator;
private:
    TokenIterator m_input_begin, m_input_end, token;
    SymbolTable m_names_table;

    std::vector<Magnum::Pointer<Function>>& m_functions;
    std::vector<Magnum::Pointer<Type>>& m_types;
    std::vector<Magnum::Pointer<Initialization>>& m_global_vars;
    std::vector<Magnum::Pointer<IndexedVariable>>& m_index_vars;
    std::vector<Magnum::Pointer<Function>> m_temp_functions;

    Magnum::Pointer<Expression> parse_expression(TokenType right_token = TokenType::Keyword_Is);
    // Helpers
    Magnum::Pointer<Variable> in_var_declaration();
    void in_return_type(Function*);
    // Parse each type of expression
    Magnum::Pointer<Expression> in_literal();
    Magnum::Pointer<Expression> in_var_expression();
    Magnum::Pointer<Expression> in_parentheses();
    Magnum::Pointer<Expression> in_expression();
    Magnum::Pointer<Expression> in_function_call();
    Magnum::Pointer<Expression> in_index_op();
    Magnum::Pointer<Expression> in_init_list();
    // Parse each type of statement
    Magnum::Pointer<Statement>       in_statement();
    Magnum::Pointer<BasicStatement>  in_basic_statement();
    Magnum::Pointer<Initialization>  in_initialization();
    Magnum::Pointer<Assignment>      in_assignment();
    Magnum::Pointer<IfBlock>         in_if_block();
    Magnum::Pointer<Block>           in_else_block();
    Magnum::Pointer<WhileLoop>       in_while_loop();
    Magnum::Pointer<ReturnStatement> in_return_statement();
    // Parse function definitions
    void in_function_definition();
    // Parse type definitions
    void in_range(multi_int& low_out, multi_int& high_out);
    void in_range_type_definition(const std::string& type_name);
    void in_array_type_definition(const std::string& type_name);
    void in_ptr_type_definition(const std::string& type_name);
    void in_type_definition();
public:
    /* Setup the parser and the related data structures. AST will
       be generated from the given token stream */
    Parser(TokenIterator input_begin, TokenIterator input_end,
           std::vector<Magnum::Pointer<Function>>&,
           std::vector<Magnum::Pointer<Type>>&,
           std::vector<Magnum::Pointer<Initialization>>& global_vars,
           std::vector<Magnum::Pointer<IndexedVariable>>&);
    /* Generate the AST */
    void run();

    friend std::ostream& operator<<(std::ostream&, const Parser&);
};
#endif
