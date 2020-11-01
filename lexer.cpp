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
#include "lexer.h"
#include <iostream>

enum class State : char {
    Start, InIdentifier, InString, InChar, InNumber, InComment
};

Lexer::Lexer(std::string::const_iterator input_begin,
             std::string::const_iterator input_end)
    : m_input_begin(input_begin),
      m_input_end(input_end),
      m_identifier_table{
         {"is", TokenType::Keyword_Is},
         {"do", TokenType::Keyword_Do},
         {"let", TokenType::Keyword_Let},
         {"if", TokenType::Keyword_If},
         {"constant", TokenType::Keyword_Const},
         {"type", TokenType::Keyword_Type},
         {"range", TokenType::Keyword_Range},
         {"function", TokenType::Keyword_Funct},
         {"end", TokenType::Keyword_End},
         {"and", TokenType::Op_And},
         {"or", TokenType::Op_Or},
         {"not", TokenType::Op_Not},
         {"thru", TokenType::Op_Thru},
         {"upto", TokenType::Op_Upto}
      }
{}

void Lexer::run()
{
    State curr_state = State::Start;
    auto input_iter = m_input_begin;
    std::string token_text;
    unsigned int line_num = 1;
    while(input_iter != m_input_end) {
        char curr = *input_iter;

        switch(curr_state) {
        case State::Start:
            if(std::isalpha(curr)) {
                // Found a keyword or a name
                curr_state = State::InIdentifier;
                token_text += curr;
            } else if(std::isdigit(curr)) {
                // Found a number literal
                curr_state = State::InNumber;
                token_text += curr;
            } else {
                switch(curr) {
                case ';':
                    m_tokens.emplace_back(line_num, TokenType::End_Statement);
                    break;
                case '"':
                    // Found start of a string literal
                    curr_state = State::InString;
                    break;
                case '\'':
                    // Found start of a char literal
                    curr_state = State::InChar;
                    break;
                case '(':
                    m_tokens.emplace_back(line_num, TokenType::Open_Parentheses);
                    break;
                case ')':
                    m_tokens.emplace_back(line_num, TokenType::Closed_Parentheses);
                    break;
                // Arithmetic operators
                case '+':
                    m_tokens.emplace_back(line_num, TokenType::Op_Plus);
                    break;
                case '-':
                    m_tokens.emplace_back(line_num, TokenType::Op_Minus);
                    break;
                case '/':
                    switch(*std::next(input_iter)) {
                    case '/':
                        ++input_iter;
                        // Ignore single-line comments
                        while(*input_iter != '\n')
                            ++input_iter;
                        break;
                    case '*':
                        // Multi-line comment
                        curr_state = State::InComment;
                        break;
                    default:
                        m_tokens.emplace_back(line_num, TokenType::Op_Div);
                        break;
                    }
                    break;
                case '*':
                    m_tokens.emplace_back(line_num, TokenType::Op_Mult);
                    break;
                case '%':
                    m_tokens.emplace_back(line_num, TokenType::Op_Mod);
                    break;
                // Bitwise operators 
                case '&':
                    m_tokens.emplace_back(line_num, TokenType::Op_Bit_And);
                    break;
                case '|':
                    m_tokens.emplace_back(line_num, TokenType::Op_Bit_Or);
                    break;
                case '^':
                    m_tokens.emplace_back(line_num, TokenType::Op_Bit_Xor);
                    break;
                case '~':
                    m_tokens.emplace_back(line_num, TokenType::Op_Bit_Not);
                    break;
                // More operators 
                case '<':
                    switch(*std::next(input_iter)) {
                    case '<':
                        // "<<"
                        m_tokens.emplace_back(line_num, TokenType::Op_Left_Shift);
                        ++input_iter;
                        break;
                    case '=':
                        // "<="
                        m_tokens.emplace_back(line_num, TokenType::Op_Le);
                        ++input_iter;
                        break;
                    default:
                        // "<"
                        m_tokens.emplace_back(line_num, TokenType::Op_Lt);
                        break;
                    }
                    break;
                case '>':
                    switch(*std::next(input_iter)) {
                    case '>':
                        // ">>"
                        m_tokens.emplace_back(line_num, TokenType::Op_Right_Shift);
                        ++input_iter;
                        break;
                    case '=':
                        // ">="
                        m_tokens.emplace_back(line_num, TokenType::Op_Ge);
                        ++input_iter;
                        break;
                    default:
                        // ">"
                        m_tokens.emplace_back(line_num, TokenType::Op_Gt);
                        break;
                    }
                    break;
                case '!':
                    if(*std::next(input_iter) == '=') {
                        m_tokens.emplace_back(line_num, TokenType::Op_Ne);
                        ++input_iter;
                    } else {
                        std::cerr << "Line " << line_num << ": Invalid operator: '!"
                                  << *std::next(input_iter)  << "'\n";
                        exit(1);
                    }
                    break;
                case ',':
                    // Comma marker (for separating arguments in functions, etc.)
                    m_tokens.emplace_back(line_num, TokenType::Comma);
                    break;
                case '=':
                    switch(*std::next(input_iter)) {
                    case '=':
                        // "=="
                        m_tokens.emplace_back(line_num, TokenType::Op_Eq);
                        ++input_iter;
                        break;
                    default:
                        // Assignment operator
                        m_tokens.emplace_back(line_num, TokenType::Op_Assign);
                        break;
                    }
                    break;
                case ':':
                    // Type indicator
                    m_tokens.emplace_back(line_num, TokenType::Type_Indicator);
                    break;
                }
            }
            break;
        case State::InIdentifier:
            // Identifiers can have letters/digits.underscores in them,
            // but not as 1st letter
            if(std::isalnum(curr) || curr == '_') {
                token_text += curr;
            } else {
                if(m_identifier_table.count(token_text) > 0) {
                    // Found a keyword
                    m_tokens.emplace_back(line_num, m_identifier_table[token_text]);
                } else {
                    // Found a name
                    m_tokens.emplace_back(line_num, TokenType::Name, token_text);
                }
                token_text.clear();
                curr_state = State::Start;
                --input_iter;
            }
            break;
        case State::InString:
            if(curr == '\\') {
                // Escape sequence
                ++input_iter;
                char escaped = escape_sequence(*input_iter);
                if(escaped == -1) {
                    std::cerr << "Error: Unrecognized escape sequence '\\"
                              << *input_iter << "'\n";
                    exit(1);
                }
                token_text += escaped;
            } else if(curr == '"') {
                // End of string
                m_tokens.emplace_back(line_num, TokenType::String_Literal, token_text);
                token_text.clear();
                curr_state = State::Start;
            } else {
                // Content of the string literal
                token_text += curr;
            }
            break;
        case State::InChar:
            if(curr == '\\') {
                // Escape sequence
                ++input_iter;
                char escaped = escape_sequence(*input_iter);
                if(escaped == -1) {
                    std::cerr << "Error: Unrecognized escape sequence '\\"
                              << *input_iter << "'\n";
                    exit(1);
                }
                curr = escaped;
            }
            token_text += curr;
            m_tokens.emplace_back(line_num, TokenType::Char_Literal, token_text);
            token_text.clear();
            ++input_iter; // Skip the ending "'" mark
            curr_state = State::Start;
            break;
        case State::InNumber:
            if(std::isdigit(curr) || curr == '.') {
                // Allow for numbers with decimal points, too
                token_text += curr;
            } else if(curr != '_') {
                // '_' is a divider; ignore, used only for readability
                // End of the digit
                if(token_text.find('.') == std::string::npos)
                    m_tokens.emplace_back(line_num, TokenType::Int_Literal, token_text);
                else
                    m_tokens.emplace_back(line_num, TokenType::Float_Literal, token_text);
                token_text.clear();
                curr_state = State::Start;
                --input_iter;
            }
            break;
        case State::InComment:
            // Ignore everything in multi-line comments
            if(curr == '*' && *std::next(input_iter) == '/') {
                ++input_iter;
                curr_state = State::Start;
            }
            break;
        }

        const char prev = *input_iter++;
        if(prev == '\n') {
            line_num++;
        }
    }

    if(!token_text.empty()) {
        std::cerr << "Incomplete token: " << token_text << '\n';
        exit(1);
    }
}

void Lexer::print_tokens()
{
    for(const auto& token : m_tokens) {
        std::cout << "Line " << token.line_num << ": " << token;
    }
}
