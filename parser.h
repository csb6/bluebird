#ifndef PARSER_CLASS_H
#define PARSER_CLASS_H
#include "token.h"
#include "ast.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <CorradePointer.h>
#include <iosfwd>

enum class NameType : char {
    LValue, Funct, DeclaredFunct,
    Type, DeclaredType
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
    std::unordered_map<std::string, NameType> m_names_table;

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
    friend std::ostream& operator<<(std::ostream&, const Parser&);
};
#endif
