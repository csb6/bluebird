#include "lexer.h"
#include <iostream>
#include <unordered_map>

std::unordered_map<std::string, TokenType> identifier_table = {
    {"is", TokenType::Keyword_Is},
    {"funct", TokenType::Keyword_Funct},
    {"ctfunct", TokenType::Keyword_Ct_Funct},
    {"end", TokenType::Keyword_End}
};

enum class State : char {
    Start, InToken, InString, InChar, InNumber
};

Lexer::Lexer(std::string::const_iterator input_begin,
             std::string::const_iterator input_end)
    : m_input_begin(input_begin),
      m_input_end(input_end)
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
                curr_state = State::InToken;
                token_text += curr;
            } else if(std::isdigit(curr)) {
                // Found a number literal
                curr_state = State::InNumber;
                token_text += curr;
            }

            switch(curr) {
            case ';':
                // Found end of a statement
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
                // Open parentheses
                m_tokens.emplace_back(line_num, TokenType::Open_Parentheses);
                break;
            case ')':
                // Closed parentheses
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
                m_tokens.emplace_back(line_num, TokenType::Op_Div);
                break;
            case '*':
                m_tokens.emplace_back(line_num, TokenType::Op_Mult);
                break;
            }
            break;
        case State::InToken:
            if(!std::isalpha(curr)) {
                if(identifier_table.count(token_text) > 0) {
                    // Found a keyword
                    m_tokens.emplace_back(line_num, identifier_table[token_text]);
                } else {
                    // Found a name
                    m_tokens.emplace_back(line_num, TokenType::Name, token_text);
                }
                token_text.clear();
                curr_state = State::Start;
                // Go back to prior char
                --input_iter;
            } else {
                token_text += curr;
            }
            break;
        case State::InString:
            if(curr == '\\') {
                // Escape sequence
                ++input_iter;
                token_text += *input_iter;
            } else if(curr != '"') {
                // Content of the string literal
                token_text += curr;
            } else {
                // End of string
                m_tokens.emplace_back(line_num, TokenType::String_Literal, token_text);
                token_text.clear();
                curr_state = State::Start;
            }
            break;
        case State::InChar:
            if(curr == '\\') {
                // Escape sequence
                ++input_iter;
                curr = *input_iter;
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
            } else if(curr != '\'') {
                // '\'' is a divider; ignore, used only for readability
                // End of the digit
                m_tokens.emplace_back(line_num, TokenType::Number_Literal, token_text);
                token_text.clear();
                curr_state = State::Start;
                --input_iter; // put the current char back
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
    }
}

void Lexer::print_tokens()
{
    for(auto[line_num, type, text] : m_tokens) {
        switch(type) {
        case TokenType::Keyword_Funct:
            std::cout << "Keyword_Funct";
            break;
        case TokenType::Keyword_Ct_Funct:
            std::cout << "Keyword_Ct_Funct";
            break;
        case TokenType::Name:
            std::cout << "Name";
            break;
        case TokenType::Keyword_Is:
            std::cout << "Keyword_Is";
            break;
        case TokenType::Open_Parentheses:
            std::cout << "Open_Paren";
            break;
        case TokenType::Closed_Parentheses:
            std::cout << "Closed_Paren";
            break;
        case TokenType::Keyword_End:
            std::cout << "Keyword_End";
            break;
        case TokenType::End_Statement:
            std::cout << "End_Statement";
            break;
        case TokenType::String_Literal:
            std::cout << "String_Literal";
            break;
        case TokenType::Char_Literal:
            std::cout << "Char_Literal";
            break;
        case TokenType::Number_Literal:
            std::cout << "Number_Literal";
            break;
        default:
            std::cout << "Unknown token type";
        }

        std::cout << "\n  Line:  " << line_num;
        std::cout << "\n  Value: " << text << '\n';
    }
}
