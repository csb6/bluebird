#include "error.h"
#include "ast.h"
#include <iostream>

#ifdef FUZZER_MODE
    Error::Error(unsigned int) {}
#else
    Error::Error(unsigned int line_num)
    {
        std::cerr << "ERROR: ";
        if(line_num > 0) {
            std::cerr << line_num << ": ";
        }
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::put(size_t) { return *this; }
#else
    Error& Error::put(size_t n)
    {
        std::cerr << n;
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::put(const char*, unsigned int) { return *this; }
#else
    Error& Error::put(const char* message, unsigned int indent)
    {
        for(; indent > 0; --indent) {
            std::cerr << " ";
        }
        std::cerr << message;
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::put(const Expression*) { return *this; }
#else
    Error& Error::put(const Expression* expr)
    {
        expr->print(std::cerr);
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::put(const Statement*) { return *this; }
#else
    Error& Error::put(const Statement* stmt)
    {
        stmt->print(std::cerr);
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::put(const Type*) { return *this; }
#else
    Error& Error::put(const Type* type)
    {
        type->print(std::cerr);
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::put(const Function*) { return *this; }
#else
    Error& Error::put(const Function* funct)
    {
        funct->print(std::cerr);
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::put(const LValue*) { return *this; }
#else
    Error& Error::put(const LValue* lval)
    {
        lval->print(std::cerr);
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::quote(const std::string&) { return *this; }
#else
    Error& Error::quote(const std::string& text)
    {
        std::cerr << " `" << text << "` ";
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::quote(char) { return *this; }
#else
    Error& Error::quote(char letter)
    {
        std::cerr << " `" << letter << "` ";
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::quote(Token) { return *this; }
#else
    Error& Error::quote(Token token)
    {
        std::cerr << " `" << token << "` ";
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::quote(TokenType) { return *this; }
#else
    Error& Error::quote(TokenType kind)
    {
        std::cerr << " `" << kind << "` ";
        return *this;
    }
#endif

#ifdef FUZZER_MODE
    Error& Error::newline() { return *this; }
#else
    Error& Error::newline()
    {
        std::cerr << "\n";
        return *this;
    }
#endif

void Error::raise(const char* message, unsigned int indent)
{
    put(message, indent);
    newline();
    exit(1);
}