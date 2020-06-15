#ifndef TOKEN_CLASS_H
#define TOKEN_CLASS_H
#include <string>
#include <iosfwd>

enum class TokenType : char {
    Keyword_Funct, Name, Keyword_Is, Keyword_Var, Open_Parentheses,
    Closed_Parentheses, Keyword_End, End_Statement, String_Literal,
    Keyword_Ct_Funct, Op_Plus, Op_Minus, Op_Div, Op_Mult, Op_Comma,
    Op_Assign, Char_Literal, Int_Literal, Float_Literal
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
    friend std::ostream& operator<<(std::ostream&, const Token&);
};

std::ostream& operator<<(std::ostream&, const TokenType);
#endif
