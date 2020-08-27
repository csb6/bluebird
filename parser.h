#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include "token.h"
#include "ast.h"
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include <CorradePointer.h>
#include <iosfwd>

struct ScopeTable {
    // Enclosing scope's index in SymbolTable::m_scopes
    short parent;
    // Most scopes won't have any types defined in them, so this field
    // is empty if set to -1. If >= 0, then is index in SymbolTable::m_discrete_types
    short discrete_type_id = -1;
    // Symbol id -> kind of symbol (e.g. lvalue, funct, etc.)
    std::unordered_map<SymbolId, NameType> symbols{};
};

class SymbolTable {
private:
    std::unordered_map<std::string, SymbolId> m_ids;
    std::vector<ScopeTable> m_scopes;
    // Contains all scopes' collections of discrete (integer-like) types
    //  For each element: maps symbol id of a type -> its range
    std::vector<std::unordered_map<SymbolId, Range>> m_discrete_types;

    short m_curr_scope;
    // Note: account for pre-defined symbol ids (e.g. for IntType, etc., see ast.h)
    SymbolId m_curr_symbol_id = FirstFreeId;
    std::optional<NameType> find_by_id(SymbolId name_id) const;
    std::pair<std::string, SymbolId> find_name(SymbolId name_id) const;
public:
    SymbolTable();
    void open_scope();
    void close_scope();
    std::optional<NameType> find(const std::string& name) const;
    // Add/update names in the current scope
    bool add(const std::string& name, NameType);
    void update(const std::string& name, NameType);
    // Checking that no names are declared but not defined (or imported)
    void validate_names();
};

class Parser {
public:
    using TokenIterator = std::vector<Token>::const_iterator;
private:
    TokenIterator m_input_begin;
    TokenIterator m_input_end;
    TokenIterator token;
    std::vector<Function> m_functions;
    std::vector<Type> m_types;
    std::vector<Magnum::Pointer<LValue>> m_lvalues;
    SymbolTable m_names_table;

    Expression* parse_expression(TokenType right_token = TokenType::Keyword_Is);
    // Helpers
    Magnum::Pointer<LValue> in_lvalue_declaration();
    // Handle each type of expression
    Expression* in_literal();
    Expression* in_lvalue_expression();
    Expression* in_parentheses();
    Expression* in_expression();
    FunctionCall* in_function_call();
    // Handle each type of statement
    Magnum::Pointer<Statement> in_statement();
    Magnum::Pointer<BasicStatement> in_basic_statement();
    Magnum::Pointer<Initialization> in_initialization();
    Magnum::Pointer<IfBlock> in_if_block();
    // Handle each function definition
    void in_function_definition();
    // Types
    void in_type_definition();
public:
    Parser(TokenIterator input_begin,
           TokenIterator input_end);
    void run();
    const auto& functions() const { return m_functions; }
    const auto& types() const { return m_types; }
    const auto& names_table() const { return m_names_table; }

    friend std::ostream& operator<<(std::ostream&, const Parser&);
};
#endif
