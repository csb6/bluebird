#ifndef TOKEN_CLASS_H
#define TOKEN_CLASS_H
#include <string>
#include <iosfwd>

enum class TokenType : char {
    // Keywords
    Keyword_Funct, Keyword_Is, Keyword_Let, Keyword_Const, Keyword_Type,
    Keyword_Ct_Funct, Keyword_End,
    // Identifiers
    Name,
    // Non-operator symbols
    Open_Parentheses, Closed_Parentheses, End_Statement, Type_Indicator,
    // Operators
    Op_Plus, Op_Minus, Op_Div, Op_Mult, Op_Comma, Op_Assign,
    // Literals
    String_Literal, Char_Literal, Int_Literal, Float_Literal
};

bool operator<(const TokenType left, const TokenType right);
bool operator>=(const TokenType left, const TokenType right);
bool operator<=(const TokenType left, const TokenType right);
bool is_operator(const TokenType);

struct Token {
    unsigned int line_num;
    TokenType type;
    std::string text;

    Token(unsigned int num, TokenType t)
        : line_num(num), type(t)
        {}
    Token(unsigned int num, TokenType t, std::string_view c)
        : line_num(num), type(t), text(c)
        {}
    friend std::ostream& operator<<(std::ostream&, const Token&);
};

std::ostream& operator<<(std::ostream&, const TokenType);
#endif
