#include "error.h"
#include "ast.h"
#include <iostream>

#ifdef FUZZER_MODE

Error::Error(unsigned int) {}
Error& Error::put(size_t) { return *this; }
Error& Error::put(const char*, unsigned int) { return *this; }
Error& Error::put(const Expression*) { return *this; }
Error& Error::put(const Statement*) { return *this; }
Error& Error::put(const Type*) { return *this; }
Error& Error::put(const Function*) { return *this; }
Error& Error::put(const Assignable*) { return *this; }
Error& Error::quote(const std::string&) { return *this; }
Error& Error::quote(char) { return *this; }
Error& Error::quote(const Token&) { return *this; }
Error& Error::quote(TokenType) { return *this; }
Error& Error::newline() { return *this; }
void Error::raise(const char*, unsigned int) { exit(1); }

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

Error& Error::put(const char* message, unsigned int indent)
{
    for(; indent > 0; --indent) {
        std::cerr << " ";
    }
    std::cerr << message;
    return *this;
}

Error& Error::put(const std::string& message, unsigned int indent)
{
    for(; indent > 0; --indent) {
        std::cerr << " ";
    }
    std::cerr << message;
    return *this;
}

Error& Error::put(const Expression* expr)
{
    expr->print(std::cerr);
    return *this;
}

Error& Error::put(const Statement* stmt)
{
    stmt->print(std::cerr);
    return *this;
}

Error& Error::put(const Type* type)
{
    type->print(std::cerr);
    return *this;
}

Error& Error::put(const Function* funct)
{
    funct->print(std::cerr);
    return *this;
}

Error& Error::put(const Assignable* assignable)
{
    assignable->print(std::cerr);
    return *this;
}

Error& Error::quote(const std::string& text)
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

void Error::raise(const char* message, unsigned int indent)
{
    put(message, indent);
    newline();
    exit(1);
}
#endif /** ifdef FUZZER_MODE */

void raise_error_expected(const char* expected, Token actual)
{
    Error(actual.line_num).put("Expected ").put(expected)
        .put(", but instead found token:\n").quote(actual).raise();
}

void raise_error_expected(const char* expected, const Expression* actual)
{
    Error(actual->line_num()).put("Expected ").put(expected)
        .put(", but instead found expression:\n")
        .put(actual).raise();
}
