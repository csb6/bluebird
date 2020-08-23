#include "lexer.h"
#include "parser.h"
#include "checker.h"
#include "codegenerator.h"
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

const std::string load_source_file(const char *filename)
{
    std::ifstream input_file(filename);
    if(!input_file) {
        std::cerr << "Error: File '" << filename << "' doesn't exist\n";
        exit(1);
    }
    return std::string{(std::istreambuf_iterator<char>(input_file)),
            (std::istreambuf_iterator<char>())};
}


int main(int argc, char **argv)
{
    if(argc < 2) {
        std::cout << "Usage: ./compiler source_file\n";
        return 1;
    }
    const std::string source_file{load_source_file(argv[1])};

    Lexer lexer{source_file.begin(), source_file.end()};
    lexer.run();
    lexer.print_tokens();

    Parser parser{lexer.begin(), lexer.end()};
    parser.run();
    
    //Checker checker{parser.functions(), parser.types()};
    //checker.run();

    //CodeGenerator codegen{parser.functions(), parser.types()};
    //codegen.run();

    std::cout << parser << '\n';

    return 0;
}
