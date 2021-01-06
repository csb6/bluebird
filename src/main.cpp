/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020  Cole Blakley

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

    Parser parser{lexer.begin(), lexer.end()};
    parser.run();

    Checker checker{parser.functions(), parser.types()};
    checker.run();

    CodeGenerator codegen{argv[1], parser.functions()};
    codegen.run();

    std::cout << parser;

    return 0;
}
