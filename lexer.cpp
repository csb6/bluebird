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
#include <ostream>
#include <array>
#include <limits>

enum class State : char {
    Start, InIdentifier, InString, InChar, InNumber, InComment
};

Lexer::Lexer(std::string::const_iterator input_begin,
             std::string::const_iterator input_end)
    : m_input_begin(input_begin),
      m_input_end(input_end),
      m_identifier_table{
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
         {"range", TokenType::Keyword_Range},
         {"and", TokenType::Op_And},
         {"or", TokenType::Op_Or},
         {"not", TokenType::Op_Not},
         {"thru", TokenType::Op_Thru},
         {"upto", TokenType::Op_Upto}
      }
{}

static void print_error(unsigned int line_num, const char *text, char token)
{
    std::cerr << "Line " << line_num << ": " << text << token << '\n';
}

static void print_error(unsigned int line_num, const char *text,
                        const std::string &token = "")
{
    std::cerr << "Line " << line_num << ": " << text << token << '\n';
}

constexpr auto state_table = []()
{
    std::array<TokenType, std::numeric_limits<unsigned char>::max()> state_table{};
    for(auto& each : state_table)
        each = TokenType::Invalid;
    constexpr char whitespace[] = " \f\n\r\t\v";
    for(unsigned char c : whitespace) {
        state_table[c] = TokenType::None;
    }

    // Identifiers
    for(unsigned char i = 'a'; i <= 'z'; ++i)
        state_table[i] = TokenType::Identifier;
    for(unsigned char i = 'A'; i <= 'Z'; ++i)
        state_table[i] = TokenType::Identifier;
    // Numbers
    for(unsigned char i = '0'; i <= '9'; ++i)
        state_table[i] = TokenType::Int_Literal;
    // Misc
    state_table['('] = TokenType::Open_Parentheses;
    state_table[')'] = TokenType::Closed_Parentheses;
    state_table[';'] = TokenType::End_Statement;
    state_table[':'] = TokenType::Type_Indicator;
    state_table[','] = TokenType::Comma;
    state_table['+'] = TokenType::Op_Plus;
    state_table['-'] = TokenType::Op_Minus;
    state_table['*'] = TokenType::Op_Mult;
    state_table['/'] = TokenType::Op_Div;
    state_table['!'] = TokenType::Op_Ne;
    state_table['<'] = TokenType::Op_Lt;
    state_table['>'] = TokenType::Op_Gt;
    state_table[','] = TokenType::Comma;
    state_table['&'] = TokenType::Op_Bit_And;
    state_table['|'] = TokenType::Op_Bit_Or;
    state_table['^'] = TokenType::Op_Bit_Xor;
    state_table['~'] = TokenType::Op_Bit_Not;
    state_table['='] = TokenType::Op_Assign;
    state_table['\''] = TokenType::Char_Literal;
    state_table['"'] = TokenType::String_Literal;
    state_table['_'] = TokenType::Underscore;

    return state_table;
}();

constexpr unsigned at(TokenType curr) { return static_cast<unsigned>(curr); }

constexpr auto transition_table = []()
{
    std::array<std::array<TokenType, TokenCount>, TokenCount> transition_table{};
    auto when = [&transition_table](TokenType curr, TokenType letter) -> auto& {
                    return transition_table[static_cast<unsigned>(curr)]
                        [static_cast<unsigned>(letter)];
                };

    for(auto& row : transition_table) {
        for(auto& each : row) {
            each = TokenType::End_Token;
        }
        row[at(TokenType::Invalid)] = TokenType::Invalid;
    }
    for(unsigned i = 0; i < TokenCount; ++i) {
        when(TokenType::None, static_cast<TokenType>(i)) = static_cast<TokenType>(i);
    }
    for(auto& each : transition_table[at(TokenType::Invalid)]) {
        each = TokenType::Invalid;
    }

    // Identifiers
    constexpr TokenType to_identifier[] = {
        TokenType::Identifier, TokenType::Int_Literal, TokenType::Underscore
    };
    for(auto each : to_identifier) {
        when(TokenType::Identifier, each) = TokenType::Identifier;
    }
    // String literals
    for(auto& each : transition_table[at(TokenType::String_Literal)]) {
        each = TokenType::String_Literal;
    }
    // End at next '"'
    when(TokenType::String_Literal, TokenType::String_Literal) = TokenType::End_Token;
    when(TokenType::String_Literal, TokenType::Invalid)        = TokenType::Invalid;
    // Char literals
    for(auto& each : transition_table[at(TokenType::Char_Literal)]) {
        each = TokenType::Char_Literal;
    }
    // End at next '\''
    when(TokenType::Char_Literal, TokenType::Char_Literal) = TokenType::End_Token;
    when(TokenType::Char_Literal, TokenType::Invalid)      = TokenType::Invalid;
    // Int literals
    when(TokenType::Int_Literal, TokenType::Int_Literal) = TokenType::Int_Literal;
    when(TokenType::Int_Literal, TokenType::Underscore)  = TokenType::Int_Literal;
    when(TokenType::Int_Literal, TokenType::Dot)         = TokenType::Float_Literal;
    // Float literals
    when(TokenType::Float_Literal, TokenType::Int_Literal) = TokenType::Float_Literal;
    // Misc
    when(TokenType::Op_Assign, TokenType::Op_Assign) = TokenType::Op_Eq;
    when(TokenType::Op_Lt,     TokenType::Op_Assign) = TokenType::Op_Le;
    when(TokenType::Op_Gt,     TokenType::Op_Assign) = TokenType::Op_Ge;
    when(TokenType::Op_Ne,     TokenType::Op_Assign) = TokenType::Op_Ne;
    when(TokenType::Op_Lt,     TokenType::Op_Lt)     = TokenType::Op_Left_Shift;
    when(TokenType::Op_Gt,     TokenType::Op_Gt)     = TokenType::Op_Right_Shift;

    return transition_table;
}();

void Lexer::process(std::istream& input)
{
    TokenType letter_state, next_state;
    TokenType token_state = TokenType::None;
    std::string token_text;
    while(input) {
        const char letter = input.get();
        letter_state = state_table[letter];
        next_state = transition_table[at(token_state)][at(letter_state)];

        switch(next_state) {
        case TokenType::End_Token:
            switch(token_state) {
            case TokenType::Op_Left_Shift:
                if(token_text.size() != 2) {
                    print_error(0, "Expected '<<', but instead found: ", token_text);
                }
                break;
            case TokenType::Op_Right_Shift:
                if(token_text.size() != 2) {
                    print_error(0, "Expected '>>', but instead found: ", token_text);
                }
                break;
            default:
                break;
            }
            m_tokens.emplace_back(0, token_state, token_text);
            token_text.clear();
            if(letter_state != TokenType::None) {
                token_text += letter;
            }
            token_state = letter_state;
            break;
        case TokenType::Invalid:
            std::cerr << "ERROR: Invalid character: '" << letter << "'\n";
            exit(1);
        case TokenType::None:
            token_text.clear();
            token_state = next_state;
            break;
        default:
            token_text += letter;
            token_state = next_state;
            break;
        }
    }
}

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
                        print_error(line_num, "Invalid operator: !");
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
                auto match = m_identifier_table.find(token_text);
                if(match != m_identifier_table.end()) {
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
                ++input_iter;
                char escaped = escape_sequence(*input_iter);
                if(escaped == -1) {
                    print_error(line_num, "Unrecognized escape sequence: \\",
                                *input_iter);
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
                    print_error(line_num, "Unrecognized escape sequence: \\",
                                *input_iter);
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
        print_error(line_num, "Incomplete token: ", token_text);
        exit(1);
    }
}

std::ostream& operator<<(std::ostream& output, const Lexer& lexer)
{
    for(const auto& token : lexer.m_tokens) {
        output << "Line " << token.line_num << ": " << token;
    }
    return output;
}
