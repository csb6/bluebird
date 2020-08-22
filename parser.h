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
    ScopeTable* parent = nullptr;
    std::unordered_map<short, NameType> symbols;
};

class SymbolTable {
private:
    // Name of symbol -> id
    std::unordered_map<std::string, short> m_ids;
    std::vector<Magnum::Pointer<ScopeTable>> m_scopes;

    ScopeTable* m_curr_scope;
    short m_curr_symbol_id = 0;
    std::optional<NameType> find_id(short name_id) const;
public:
    SymbolTable();
    void open_scope();
    void close_scope();
    std::optional<NameType> find(const std::string& name) const;
    bool add(const std::string& name, NameType);
    void update(const std::string& name, NameType);
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
    void validate_names() const;
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
