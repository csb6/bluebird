#include "lexer.h"
#include "parser.h"
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

const std::string load_source_file(const char *filename)
{
    std::ifstream input_file(filename);
    return std::string{(std::istreambuf_iterator<char>(input_file)),
            (std::istreambuf_iterator<char>())};
}


int main()
{
    const std::string source_file{load_source_file("parser-test.txt")};

    Lexer lexer{source_file.begin(), source_file.end()};
    lexer.run();
    lexer.print_tokens();

    Parser parser{lexer.begin(), lexer.end()};
    parser.run();

    std::cout << parser << '\n';

    return 0;
}
