/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2022  Cole Blakley

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
#include "error.h"
#include "lexer.h"
#include <ostream>
#include <frozen/unordered_map.h>
#include <frozen/string.h>

#ifdef FUZZER_MODE
/* ret_val is the return type of the containing function (don't pass an argument
   if returning void) */
  #define fatal_error(ret_val) return ret_val
#else
  #include <cstdlib>
  #define fatal_error(ret_val) (exit(1))
#endif

static inline
bool check_valid_next_exists(unsigned int line_num, std::string::const_iterator curr, std::string::const_iterator end)
{
    if(std::next(curr) == end) {
        Error(line_num).put("Unexpected end of file\n");
        return false;
    }
    return true;
}

enum class State : char {
    Start, InIdentifier, InString, InChar, InNumber, InComment
};

constexpr static auto identifier_table = frozen::make_unordered_map<frozen::string, TokenType>(
    {
        {"function", TokenType::Keyword_Funct},
        {"is", TokenType::Keyword_Is},
        {"do", TokenType::Keyword_Do},
        {"let", TokenType::Keyword_Let},
        {"constant", TokenType::Keyword_Const},
        {"type", TokenType::Keyword_Type},
        {"end", TokenType::Keyword_End},
        {"mod", TokenType::Op_Mod},
        {"rem", TokenType::Op_Rem},
        {"if", TokenType::Keyword_If},
        {"else", TokenType::Keyword_Else},
        {"while", TokenType::Keyword_While},
        {"range", TokenType::Keyword_Range},
        {"return", TokenType::Keyword_Return},
        {"and", TokenType::Op_And},
        {"or", TokenType::Op_Or},
        {"xor", TokenType::Op_Xor},
        {"not", TokenType::Op_Not},
        {"thru", TokenType::Op_Thru},
        {"upto", TokenType::Op_Upto},
        {"to_val", TokenType::Op_To_Val},
        {"to_ptr", TokenType::Op_To_Ptr},
        {"true", TokenType::Keyword_True},
        {"false", TokenType::Keyword_False},
        {"of", TokenType::Keyword_Of},
        {"loop", TokenType::Keyword_Loop},
        {"ptr", TokenType::Keyword_Ptr}
    });

Lexer::Lexer(std::string::const_iterator input_begin,
             std::string::const_iterator input_end)
    : m_input_begin(input_begin), m_input_end(input_end) {}

void Lexer::run()
{
    State curr_state = State::Start;
    auto input_iter = m_input_begin;
    std::string token_text;
    unsigned int line_num = 1;
    while(input_iter < m_input_end) {
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
                case '[':
                    m_tokens.emplace_back(line_num, TokenType::Open_Bracket);
                    break;
                case ']':
                    m_tokens.emplace_back(line_num, TokenType::Closed_Bracket);
                    break;
                case '{':
                    m_tokens.emplace_back(line_num, TokenType::Open_Curly);
                    break;
                case '}':
                    m_tokens.emplace_back(line_num, TokenType::Closed_Curly);
                    break;
                // Arithmetic operators
                case '+':
                    m_tokens.emplace_back(line_num, TokenType::Op_Plus);
                    break;
                case '-':
                    m_tokens.emplace_back(line_num, TokenType::Op_Minus);
                    break;
                case '/':
                    if(!check_valid_next_exists(line_num, input_iter, m_input_end)) {
                        fatal_error();
                    }
                    switch(*std::next(input_iter)) {
                    case '/':
                        ++input_iter;
                        // Ignore single-line comments
                        while(input_iter < m_input_end && *input_iter != '\n')
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
                // More operators
                case '<':
                    if(!check_valid_next_exists(line_num, input_iter, m_input_end)) {
                        fatal_error();
                    }
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
                    if(!check_valid_next_exists(line_num, input_iter, m_input_end)) {
                        fatal_error();
                    }
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
                    if(!check_valid_next_exists(line_num, input_iter, m_input_end)) {
                        fatal_error();
                    }
                    if(*std::next(input_iter) == '=') {
                        // "!="
                        m_tokens.emplace_back(line_num, TokenType::Op_Ne);
                        ++input_iter;
                    } else {
                        Error(line_num).put("Invalid operator: `!`\n");
                        fatal_error();
                    }
                    break;
                case ',':
                    m_tokens.emplace_back(line_num, TokenType::Comma);
                    break;
                case '=':
                    m_tokens.emplace_back(line_num, TokenType::Op_Eq);
                    break;
                case ':':
                    if(!check_valid_next_exists(line_num, input_iter, m_input_end)) {
                        fatal_error();
                    }
                    switch(*std::next(input_iter)) {
                    case '=':
                        m_tokens.emplace_back(line_num, TokenType::Op_Assign);
                        ++input_iter;
                        break;
                    default:
                        m_tokens.emplace_back(line_num, TokenType::Type_Indicator);
                        break;
                    }
                }
            }
            break;
        case State::InIdentifier:
            // Identifiers can have letters/digits.underscores in them,
            // but not as 1st letter
            if(std::isalnum(curr) || curr == '_') {
                token_text += curr;
            } else {
                auto match = identifier_table.find(std::string_view{token_text});
                if(match != identifier_table.end()) {
                    // Found a keyword
                    m_tokens.emplace_back(line_num, match->second);
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
                if(!check_valid_next_exists(line_num, input_iter, m_input_end)) {
                    fatal_error();
                }
                ++input_iter;
                char escaped = escape_sequence(*input_iter);
                if(escaped == -1) {
                    Error(line_num)
                        .put("Unrecognized escape sequence:")
                        .quote(*input_iter).newline();
                    fatal_error();
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
                if(!check_valid_next_exists(line_num, input_iter, m_input_end)) {
                    fatal_error();
                }
                ++input_iter;
                char escaped = escape_sequence(*input_iter);
                if(escaped == -1) {
                    Error(line_num)
                        .put("Unrecognized escape sequence:")
                        .quote(*input_iter).newline();
                    fatal_error();
                }
                curr = escaped;
            }
            token_text += curr;
            m_tokens.emplace_back(line_num, TokenType::Char_Literal, token_text);
            token_text.clear();
            if(!check_valid_next_exists(line_num, input_iter, m_input_end)) {
                fatal_error();
            }
            ++input_iter;
            if(*input_iter != '\'') {
                Error(line_num).put("Expected closing single-quote at end of character literal\n");
                fatal_error();
            }
            curr_state = State::Start;
            break;
        case State::InNumber:
            if(std::isdigit(curr) || curr == '.') {
                // Allow for numbers with decimal points, too
                token_text += curr;
            } else if(curr != '_') {
                // '_' is a divider; ignore, used only for readability
                // End of the digit
                if(token_text.find('.') == std::string::npos) {
                    m_tokens.emplace_back(line_num, TokenType::Int_Literal, token_text);
                } else {
                    m_tokens.emplace_back(line_num, TokenType::Float_Literal, token_text);
                }
                token_text.clear();
                curr_state = State::Start;
                --input_iter;
            }
            break;
        case State::InComment:
            if(!check_valid_next_exists(line_num, input_iter, m_input_end)) {
                fatal_error();
            }
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
        Error(line_num)
            .put("Incomplete token:").quote(token_text).newline();
        fatal_error();
    }
}

std::ostream& operator<<(std::ostream& output, const Lexer& lexer)
{
    for(const auto& token : lexer.m_tokens) {
        output << "Line " << token.line_num << ": " << token;
    }
    return output;
}
