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
    Comma,
    // Operators
    Op_Plus, Op_Minus, Op_Div, Op_Mult, Op_Assign,
    // Literals
    String_Literal, Char_Literal, Int_Literal, Float_Literal
};

using Precedence = char;
constexpr Precedence Invalid_Operator = -2;
constexpr Precedence Literal_Token = 100;

constexpr Precedence operator_precedence_table[] = {
    // Keywords
    //  Keyword_Funct:
         Invalid_Operator,
    //  Keyword_Is:
         Invalid_Operator,
    //  Keyword_Let:
         Invalid_Operator,
    //  Keyword_Const:
         Invalid_Operator,
    //  Keyword_Type:
         Invalid_Operator,
    //  Keyword_Ct_Funct:
         Invalid_Operator,
    //  Keyword_End:
         Invalid_Operator,
    // Identifiers
    //  Name:
         Invalid_Operator,
    // Non-operator symbols
    //  Open_Parentheses:
         Invalid_Operator,
    //  Closed_Parentheses:
         Invalid_Operator,
    //  End_Statement:
         Invalid_Operator,
    //  Type_Indicator:
         Invalid_Operator,
    //  Comma:
         Invalid_Operator,
    // Operators
    //  Op_Plus:
         0,
    //  Op_Minus:
         0,
    //  Op_Div:
         1,
    //  Op_Mult:
         1,
    //  Op_Assign:
         2,
    // Literals
    //  String_Literal:
         Literal_Token,
    //  Char_Literal:
         Literal_Token,
    //  Int_Literal:
         Literal_Token,
    //  Float_Literal:
         Literal_Token
};

static_assert(sizeof(operator_precedence_table) / sizeof(operator_precedence_table[0]) == 22, "Table out-of-sync with TokenType enum");

constexpr Precedence precedence_of(const TokenType index)
{
    return operator_precedence_table[int(index)];
}

constexpr bool is_operator(const TokenType token)
{
    const Precedence p = precedence_of(token);
    return p != Invalid_Operator && p != Literal_Token;
}

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
