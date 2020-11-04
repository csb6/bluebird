/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020  Cole Blakley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef TOKEN_CLASS_H
#define TOKEN_CLASS_H
#include <string>
#include <iosfwd>

enum class TokenType : char {
    // Keywords
    Keyword_Funct, Keyword_Is, Keyword_Do, Keyword_Let, Keyword_Const, Keyword_Type,
    Keyword_End, Keyword_If, Keyword_Else, Keyword_Range,
    // Non-operator symbols
    Open_Parentheses, Closed_Parentheses, End_Statement, Type_Indicator,
    Comma,
    // Operators
    //  Arithmetic
    Op_Plus, Op_Minus, Op_Div, Op_Mult, Op_Mod, Op_Rem,
    //  Logical
    Op_And, Op_Or, Op_Not,
    //  Comparison
    Op_Eq, Op_Ne, Op_Lt, Op_Gt, Op_Le, Op_Ge,
    //  Bitwise
    Op_Left_Shift, Op_Right_Shift, Op_Bit_And, Op_Bit_Or, Op_Bit_Xor, Op_Bit_Not,
    //  Range
    Op_Thru, Op_Upto,
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

char escape_sequence(char);
void print_unescape(char source, std::ostream&);
void print_unescape(const std::string& source, std::ostream&);
#endif
