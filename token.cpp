#include "token.h"
#include <ostream>

std::ostream& operator<<(std::ostream& output, const TokenType type)
{
    switch(type) {
    case TokenType::Keyword_Funct:
        output << "Keyword `function`";
        break;
    case TokenType::Name:
        output << "Name";
        break;
    case TokenType::Keyword_Is:
        output << "Keyword `is`";
        break;
    case TokenType::Keyword_Let:
        output << "Keyword `let`";
        break;
    case TokenType::Keyword_Const:
        output << "Keyword `constant`";
        break;
    case TokenType::Keyword_Type:
        output << "Keyword `type`";
        break;
    case TokenType::Open_Parentheses:
        output << "Open Parentheses";
        break;
    case TokenType::Closed_Parentheses:
        output << "Closed Parentheses";
        break;
    case TokenType::Keyword_End:
        output << "Keyword `end`";
        break;
    case TokenType::End_Statement:
        output << "End Statement Marker (a.k.a `;`)";
        break;
    case TokenType::String_Literal:
        output << "String Literal";
        break;
    case TokenType::Keyword_Ct_Funct:
        output << "Keyword `ct_funct`";
        break;
    case TokenType::Op_Plus:
        output << "Addition operator";
        break;
    case TokenType::Op_Minus:
        output << "Subtraction operator";
        break;
    case TokenType::Op_Div:
        output << "Division operator";
        break;
    case TokenType::Op_Mult:
        output << "Multiplication operator";
        break;
    case TokenType::Comma:
        output << "Comma marker";
        break;
    case TokenType::Op_Assign:
        output << "Assignment operator";
        break;
    case TokenType::Type_Indicator:
        output << "Type Indicator";
        break;
    case TokenType::Char_Literal:
        output << "Character Literal";
        break;
    case TokenType::Int_Literal:
        output << "Integer Literal";
        break;
    case TokenType::Float_Literal:
        output << "Float Literal";
        break;
    }

    return output;
}

std::ostream& operator<<(std::ostream& output, const Token& token)
{
    output << token.type << '\n';

    if(!token.text.empty()) {
        output << "  Text: " << token.text << '\n';
    }
    return output;
}
