#ifndef LEXER_CLASS_H
#define LEXER_CLASS_H
#include <string>
#include <vector>
#include <unordered_map>
#include "token.h"

class Lexer {
private:
    std::vector<Token> m_tokens;
    std::string::const_iterator m_input_begin;
    std::string::const_iterator m_input_end;
    std::unordered_map<std::string, TokenType> m_identifier_table;
public:
    explicit Lexer(std::string::const_iterator input_begin,
                   std::string::const_iterator input_end);
    void run();
    void print_tokens();
    // Iterators for accessing the tokens
    auto begin() { return m_tokens.begin(); }
    auto end() { return m_tokens.end(); }
    auto begin() const { return m_tokens.begin(); }
    auto end() const { return m_tokens.end(); }
};
#endif
