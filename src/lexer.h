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
#ifndef LEXER_CLASS_H
#define LEXER_CLASS_H
#include <vector>
#include <unordered_map>
#include "token.h"

class Lexer {
private:
    std::vector<Token> m_tokens;
    std::string::const_iterator m_input_begin;
    std::string::const_iterator m_input_end;
    std::unordered_map<std::string, TokenType> m_identifier_table;

    friend std::ostream& operator<<(std::ostream&, const Lexer&);
public:
    explicit Lexer(std::string::const_iterator input_begin,
                   std::string::const_iterator input_end);
    void run();
    void print_tokens();
    // Iterators for accessing the tokens
    auto begin() { return m_tokens.begin(); }
    auto end() { return m_tokens.end(); }
    auto begin() const { return m_tokens.begin(); }
    auto end() const { return m_tokens.end(); }
};
#endif
