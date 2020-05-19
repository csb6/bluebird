#ifndef TOKEN_CLASS_H
#define TOKEN_CLASS_H
#include <string>

enum class TokenType : char {
    Keyword_Funct, Name, Keyword_Is, Open_Parentheses,
    Closed_Parentheses, Keyword_End, End_Statement, String_Literal,
    Keyword_Ct_Funct, Op_Plus, Op_Minus, Op_Div, Op_Mult, Char_Literal,
    Number_Literal
};

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
};
#endif
