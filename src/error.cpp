#include "error.h"
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
    Error& Error::newline() { return *this; }
#else
    Error& Error::newline()
    {
        std::cerr << "\n";
        return *this;
    }
#endif

void Error::end()
{
    newline();
    raise();
}

void Error::raise()
{
    exit(1);
}

void Error::put_end(const char* message, unsigned int indent)
{
    put(message, indent);
    newline();
    raise();
}
