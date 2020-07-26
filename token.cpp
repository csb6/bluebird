#include "token.h"
#include <ostream>

std::ostream& operator<<(std::ostream& output, const TokenType type)
{
    switch(type) {
    case TokenType::Keyword_Funct:
        output << "Keyword `funct`\n";
        break;
    case TokenType::Name:
        output << "Name\n";
        break;
    case TokenType::Keyword_Is:
        output << "Keyword `is`\n";
        break;
    case TokenType::Keyword_Let:
        output << "Keyword `let`\n";
        break;
    case TokenType::Open_Parentheses:
        output << "Open Parentheses\n";
        break;
    case TokenType::Closed_Parentheses:
        output << "Closed Parentheses\n";
        break;
    case TokenType::Keyword_End:
        output << "Keyword `end`\n";
        break;
    case TokenType::End_Statement:
        output << "End Statement Marker (a.k.a `;`)\n";
        break;
    case TokenType::String_Literal:
        output << "String Literal\n";
        break;
    case TokenType::Keyword_Ct_Funct:
        output << "Keyword `ct_funct`\n";
        break;
    case TokenType::Op_Plus:
        output << "Addition operator\n";
        break;
    case TokenType::Op_Minus:
        output << "Subtraction operator\n";
        break;
    case TokenType::Op_Div:
        output << "Division operator\n";
        break;
    case TokenType::Op_Mult:
        output << "Multiplication operator\n";
        break;
    case TokenType::Op_Comma:
        output << "Comma marker\n";
        break;
    case TokenType::Op_Assign:
        output << "Assignment operator\n";
        break;
    case TokenType::Type_Indicator:
        output << "Type Indicato\n";
        break;
    case TokenType::Char_Literal:
        output << "Character Literal\n";
        break;
    case TokenType::Int_Literal:
        output << "Integer Literal\n";
        break;
    case TokenType::Float_Literal:
        output << "Float Literal\n";
        break;
    }

    return output;
}

std::ostream& operator<<(std::ostream& output, const Token& token)
{
    output << token.type;

    if(!token.text.empty()) {
        output << "  Text: " << token.text << '\n';
    }
    return output;
}
