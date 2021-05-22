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
#include "token.h"
#include <ostream>

std::ostream& operator<<(std::ostream& output, const TokenType type)
{
    switch(type) {
    case TokenType::Keyword_Funct:
        output << "Keyword `function` ";
        break;
    case TokenType::Name:
        output << "Name";
        break;
    case TokenType::Keyword_Is:
        output << "Keyword `is` ";
        break;
    case TokenType::Keyword_Do:
        output << "Keyword `do` ";
        break;
    case TokenType::Keyword_Let:
        output << "Keyword `let` ";
        break;
    case TokenType::Keyword_Const:
        output << "Keyword `constant` ";
        break;
    case TokenType::Keyword_Type:
        output << "Keyword `type` ";
        break;
    case TokenType::Keyword_Of:
        output << "Keyword `of` ";
        break;
    case TokenType::Keyword_Loop:
        output << "Keyword `loop` ";
        break;
    case TokenType::Keyword_Ref:
        output << "Keyword `ref` ";
        break;
    case TokenType::Open_Parentheses:
        output << "Open Parentheses";
        break;
    case TokenType::Closed_Parentheses:
        output << "Closed Parentheses";
        break;
    case TokenType::Keyword_End:
        output << "Keyword `end` ";
        break;
    case TokenType::Keyword_If:
        output << "Keyword `if` ";
        break;
    case TokenType::Keyword_Else:
        output << "Keyword `else` ";
        break;
    case TokenType::Keyword_While:
        output << "Keyword `while` ";
        break;
    case TokenType::Keyword_Range:
        output << "Keyword `range` ";
        break;
    case TokenType::Keyword_Return:
        output << "Keyword `return` ";
        break;
    case TokenType::Keyword_True:
        output << "Keyword `true` ";
        break;
    case TokenType::Keyword_False:
        output << "Keyword `false` ";
        break;
    case TokenType::End_Statement:
        output << "End Statement (a.k.a `;`) ";
        break;
    case TokenType::String_Literal:
        output << "String Literal";
        break;
    case TokenType::Op_Plus:
        output << '+';
        break;
    case TokenType::Op_Minus:
        output << '-';
        break;
    case TokenType::Op_Div:
        output << '/';
        break;
    case TokenType::Op_Mult:
        output << '*';
        break;
    case TokenType::Op_Mod:
        output << "mod";
        break;
    case TokenType::Op_Rem:
        output << "rem";
        break;
    case TokenType::Op_And:
        output << "and";
        break;
    case TokenType::Op_Or:
        output << "or";
        break;
    case TokenType::Op_Not:
        output << "not";
        break;
    case TokenType::Op_Eq:
        output << "=";
        break;
    case TokenType::Op_Ne:
        output << "!=";
        break;
    case TokenType::Op_Lt:
        output << '<';
        break;
    case TokenType::Op_Gt:
        output << '>';
        break;
    case TokenType::Op_Le:
        output << "<=";
        break;
    case TokenType::Op_Ge:
        output << ">=";
        break;
    case TokenType::Op_Left_Shift:
        output << "<<";
        break;
    case TokenType::Op_Right_Shift:
        output << ">>";
        break;
    case TokenType::Op_Thru:
        output << "thru";
        break;
    case TokenType::Op_Upto:
        output << "upto";
        break;
    case TokenType::Comma:
        output << "Comma marker";
        break;
    case TokenType::Open_Bracket:
        output << "Open bracket";
        break;
    case TokenType::Closed_Bracket:
        output << "Closed bracket";
        break;
    case TokenType::Open_Curly:
        output << "Open curly bracket";
        break;
    case TokenType::Closed_Curly:
        output << "Closed curly bracket";
        break;
    case TokenType::Op_Assign:
        output << "Assignment operator";
        break;
    case TokenType::Type_Indicator:
        output << "Type indicator";
        break;
    case TokenType::Char_Literal:
        output << "Character literal";
        break;
    case TokenType::Int_Literal:
        output << "Integer literal";
        break;
    case TokenType::Float_Literal:
        output << "Float literal";
        break;
    }

    return output;
}

std::ostream& operator<<(std::ostream& output, const Token& token)
{
    output << token.type;

    if(!token.text.empty()) {
        output << "\n  Text: ";
        print_unescape(token.text, output);
    }
    return output;
}

char escape_sequence(char letter)
{
    switch(letter) {
    case 'a': return '\a';
    case 'b': return '\b';
    case 'f': return '\f';
    case 'n': return '\n';
    case 'r': return '\r';
    case 't': return '\t';
    case 'v': return '\v';
    case '\\':
    case '\'':
    case '"':
        return letter;
    default:
        // Invalid escape sequence
        return -1;
    }
}

void print_unescape(char source, std::ostream& output)
{
    switch(source) {
    case '\a':
        output << "\\a";
        break;
    case '\b':
        output << "\\b";
        break;
    case '\f':
        output << "\\f";
        break;
    case '\n':
        output << "\\n";
        break;
    case '\r':
        output << "\\r";
        break;
    case '\t':
        output << "\\t";
        break;
    case '\v':
        output << "\\v";
        break;
    default:
        output << source;
    }
}

void print_unescape(const std::string& source, std::ostream& output)
{
    for(char letter : source) {
        print_unescape(letter, output);
    }
}

bool is_bool_op(const TokenType token)
{
    return token >= TokenType::Op_And && token <= TokenType::Op_Ge;
}
