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

struct SymbolInfo {
    NameType name_type;
    size_t index = -1;
};

struct Scope {
    short parent_index;
    std::unordered_map<std::string, SymbolInfo> symbols{};
};

class SymbolTable {
public:
    SymbolTable(std::vector<RangeType>& range_types,
                std::vector<Magnum::Pointer<LValue>>& lvalues,
                std::vector<Function>& functions);
    void open_scope();
    void close_scope();
    std::optional<SymbolInfo> find(const std::string& name) const;

    const RangeType& get_range_type(short index) const;
    const LValue& get_lvalue(short index) const;
    const Function& get_function(short index) const;

    // All add functions assume the name isn't already used for something else
    void add_lvalue(Magnum::Pointer<LValue>);
    void add_type(RangeType&&);
    void add_type(const std::string& name);
    void add_function(Function&&);
    void add_function(const std::string& name);

    // Checking that no names are declared but not defined (or imported)
    void validate_names();
private:
    // Scope tree
    std::vector<Scope> m_scopes;
    short m_curr_scope;

    // AST entities
    std::vector<RangeType>& m_range_types;
    std::vector<Magnum::Pointer<LValue>>& m_lvalues;
    std::vector<Function>& m_functions;

    SymbolId m_curr_symbol_id;
};

class Parser {
public:
    using TokenIterator = std::vector<Token>::const_iterator;
private:
    TokenIterator m_input_begin;
    TokenIterator m_input_end;
    TokenIterator token;

    std::vector<RangeType> m_range_types;
    std::vector<Magnum::Pointer<LValue>> m_lvalues;
    std::vector<Function> m_functions;
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
    void in_range_type_definition(const std::string& type_name);
    void in_type_definition();
public:
    Parser(TokenIterator input_begin,
           TokenIterator input_end);
    void run();
    const auto& functions() const { return m_functions; }
    const auto& types() const { return m_range_types; }
    const auto& names_table() const { return m_names_table; }

    friend std::ostream& operator<<(std::ostream&, const Parser&);
};
#endif
