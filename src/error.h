#ifndef BLUEBIRD_ERROR_H
#define BLUEBIRD_ERROR_H
/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2021  Cole Blakley

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
#include "token.h"

class Error {
public:
    explicit Error(unsigned int line_num = 0);
    Error& quote(const std::string& text);
    Error& quote(char);
    Error& quote(Token);
    Error& quote(TokenType);
    Error& put(size_t);
    Error& put(const char* message, unsigned int indent = 0);
    Error& put(const std::string& message, unsigned int indent = 0);
    Error& put(const struct Expression*);
    Error& put(const struct Statement*);
    Error& put(const struct Type*);
    Error& put(const struct Function*);
    Error& put(const struct Assignable*);
    Error& newline();
    // Exits or throws exception; newline printed after message
    [[noreturn]] void raise(const char* message = "", unsigned int indent = 0);
};

[[noreturn]] void raise_error_expected(const char* expected, Token actual);
[[noreturn]] void raise_error_expected(const char* expected, const Expression* actual);
#endif
