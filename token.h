#ifndef TOKEN_CLASS_H
#define TOKEN_CLASS_H
#include <string>
#include <iosfwd>

enum class TokenType : char {
    // Keywords
    Keyword_Funct, Keyword_Is, Keyword_Do, Keyword_Let, Keyword_Const, Keyword_Type,
    Keyword_Ct_Funct, Keyword_End, Keyword_If, Keyword_Else,
    // Non-operator symbols
    Open_Parentheses, Closed_Parentheses, End_Statement, Type_Indicator,
    Comma,
    // Operators
    //  Arithmetic
    Op_Plus, Op_Minus, Op_Div, Op_Mult, Op_Mod,
    //  Logical
    Op_And, Op_Or,
    //  Comparison
    Op_Eq, Op_Ne, Op_Lt, Op_Gt, Op_Le, Op_Ge,
    //  Bitwise
    Op_Left_Shift, Op_Right_Shift, Op_Bit_And, Op_Bit_Or, Op_Bit_Xor,
    // Pseudo-Operators (like operators in appearance, but not evaluated in the Pratt
    //  parser code and not subject to precedence rules/table)
    Op_Assign,
    // Operands
    String_Literal, Char_Literal, Int_Literal, Float_Literal, /*End marker:*/ Name
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
