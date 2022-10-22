/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2021-2022  Cole Blakley

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
#include "ast.h"
#include "astprinter.h"
#include <iostream>

#ifdef FUZZER_MODE

Error::Error(unsigned int) {}
Error& Error::put(size_t) { return *this; }
Error& Error::put(std::string_view, unsigned int) { return *this; }
Error& Error::put(const Expression*) { return *this; }
Error& Error::put(const Statement*) { return *this; }
Error& Error::put(const Type*) { return *this; }
Error& Error::put(const Function*) { return *this; }
Error& Error::put(const Assignable*) { return *this; }
Error& Error::quote(std::string_view) { return *this; }
Error& Error::quote(char) { return *this; }
Error& Error::quote(const Token&) { return *this; }
Error& Error::quote(TokenType) { return *this; }
Error& Error::newline() { return *this; }
void Error::raise(std::string_view, unsigned int) { exit(1); }

#else

Error::Error(unsigned int line_num)
{
    std::cerr << "ERROR: ";
    if(line_num > 0) {
        std::cerr << "Line " << line_num << ": ";
    }
}

Error& Error::put(size_t n)
{
    std::cerr << n;
    return *this;
}

Error& Error::put(std::string_view message, unsigned int indent)
{
    for(unsigned int i = 0; i < indent; ++i) {
        std::cerr << " ";
    }
    std::cerr << message;
    return *this;
}

Error& Error::put(const Expression* expr)
{
    std::cerr << *expr;
    return *this;
}

Error& Error::put(const Statement* stmt)
{
    std::cerr << *stmt;
    return *this;
}

Error& Error::put(const Type* type)
{
    std::cerr << "Type: " << type->name;
    return *this;
}

Error& Error::put(const Function* funct)
{
    std::cerr << *funct;
    return *this;
}

Error& Error::put(const Assignable* assignable)
{
    std::cerr << *assignable;
    return *this;
}

Error& Error::quote(std::string_view text)
{
    std::cerr << " `" << text << "` ";
    return *this;
}

Error& Error::quote(char letter)
{
    std::cerr << " `" << letter << "` ";
    return *this;
}

Error& Error::quote(const Token& token)
{
    std::cerr << " `" << token << "` ";
    return *this;
}

Error& Error::quote(TokenType kind)
{
    std::cerr << " `" << kind << "` ";
    return *this;
}

Error& Error::newline()
{
    std::cerr << "\n";
    return *this;
}

void Error::raise(std::string_view message, unsigned int indent)
{
    put(message, indent);
    newline();
    exit(1);
}
#endif /** ifdef FUZZER_MODE */

void raise_error_expected(std::string_view expected, const Token& actual)
{
    Error(actual.line_num).put("Expected ").put(expected)
        .put(", but instead found token:\n").quote(actual).raise();
}

void raise_error_expected(std::string_view expected, const Expression* actual)
{
    Error(actual->line_num()).put("Expected ").put(expected)
        .put(", but instead found expression:\n")
        .put(actual).raise();
}

void unreachable(std::string_view message, std::string_view filename, unsigned int line_num)
{
    Error().put(filename).put(":").put(std::to_string(line_num)).put(": ").raise(message);
}
