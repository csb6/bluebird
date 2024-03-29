#ifndef TOKEN_CLASS_H
#define TOKEN_CLASS_H
/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2021  Cole Blakley

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
#include <string>
#include <iosfwd>

enum class TokenType : unsigned char {
    // Keywords
    Keyword_Funct, Keyword_Is, Keyword_Do, Keyword_Let, Keyword_Const, Keyword_Type,
    Keyword_End, Keyword_If, Keyword_Else, Keyword_While, Keyword_Range, Keyword_Return,
    Keyword_True, Keyword_False, Keyword_Of, Keyword_Loop, Keyword_Ref, Keyword_Ptr,
    // Non-operator symbols
    Open_Parentheses, Closed_Parentheses, End_Statement, Type_Indicator,
    Comma, Open_Bracket, Closed_Bracket, Open_Curly, Closed_Curly,
    // Operators
    //  Arithmetic
    Op_Plus, Op_Minus, Op_Div, Op_Mult, Op_Mod, Op_Rem,
    //  Logical
    /*bool_op_start is Op_And*/ Op_And, Op_Or, Op_Not, Op_Xor,
    //  Comparison
    Op_Eq, Op_Ne, Op_Lt, Op_Gt, Op_Le, Op_Ge, /*bool_op_end is Op_Ge*/
    //  Bitwise
    Op_Left_Shift, Op_Right_Shift,
    //  Range
    Op_Thru, Op_Upto,
    //  Pointer
    Op_To_Val, Op_To_Ptr,
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
bool is_bool_op(const TokenType);
bool is_ptr_op(const TokenType);
#endif
