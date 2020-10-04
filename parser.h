#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include "token.h"
#include "ast.h"
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include <CorradePointer.h>
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
    short parent_index;
    std::unordered_map<std::string, SymbolInfo> symbols{};
};

class SymbolTable {
public:
    SymbolTable(MemoryPool& range_types,
                MemoryPool& lvalues,
                MemoryPool& functions);
    void open_scope();
    void close_scope();
    std::optional<SymbolInfo> find(const std::string& name) const;

    // All add functions assume the name isn't already used for something else
    LValue* add_lvalue(LValue&&);
    RangeType* add_type(RangeType&&);
    RangeType* add_type(const std::string& name);
    Function* add_function(Function&&);
    Function* add_function(const std::string& name);

    // Checking that no names are declared but not defined (or imported)
    void validate_names();
private:
    // Scope tree
    std::vector<Scope> m_scopes;
    short m_curr_scope;

    // AST entities
    MemoryPool &m_range_types;
    MemoryPool &m_lvalues;
    MemoryPool &m_functions;

    SymbolId m_curr_symbol_id;
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

    Expression* parse_expression(TokenType right_token = TokenType::Keyword_Is);
    // Helpers
    LValue in_lvalue_declaration();
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
    IfBlock* in_if_block();
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
