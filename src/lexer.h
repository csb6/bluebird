#ifndef LEXER_CLASS_H
#define LEXER_CLASS_H
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
#include <vector>
#include "token.h"

/* This file contains a class that transforms a stream of characters into
   tokens. It checks for invalid or misplaced tokens as it generates them.
   Keywords and identifier names will be distinguished in this stage, as both
   are separate kinds of tokens. */

class Lexer {
private:
    std::vector<Token> m_tokens;
    std::string::const_iterator m_input_begin;
    std::string::const_iterator m_input_end;
public:
    /* Setup the lexer/its data structures */
    Lexer(std::string::const_iterator input_begin,
          std::string::const_iterator input_end);
    /* Run the lexer of the given character stream */
    void run();
    // Iterators for accessing the tokens
    auto begin() { return m_tokens.begin(); }
    auto end() { return m_tokens.end(); }
    auto begin() const { return m_tokens.begin(); }
    auto end() const { return m_tokens.end(); }

    friend std::ostream& operator<<(std::ostream&, const Lexer&);
};
#endif
