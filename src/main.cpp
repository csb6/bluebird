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
        std::cerr << "Usage: ./compiler [-g | --debug] source_file\n";
        return 1;
    }
    CodeGenerator::Mode build_mode = CodeGenerator::Mode::Default;
    int arg_index = 1;
    while(argv[arg_index][0] == '-') {
        if(strcmp(argv[arg_index], "--debug") == 0
           || strcmp(argv[arg_index], "-g") == 0) {
            // Build executables with debug info
            build_mode = CodeGenerator::Mode::Debug;
            ++arg_index;
        } else if(strcmp(argv[arg_index], "--optimize") == 0
                  || strcmp(argv[arg_index], "-O") == 0) {
            // Build executables with optimizations enabled
            build_mode = CodeGenerator::Mode::Optimize;
            ++arg_index;
        } else {
            std::cerr << "Error: unknown option '" << argv[arg_index] << "'\n";
            return 1;
        }
        if(arg_index >= argc) {
            std::cerr << "Error: Missing source filename\n";
            return 1;
        }
    }
    const char* source_filename = argv[arg_index];
    const std::string source_file{load_source_file(source_filename)};

    Lexer lexer{source_file.begin(), source_file.end()};
    lexer.run();

    Parser parser{lexer.begin(), lexer.end()};
    parser.run();

    Checker checker{parser.functions(), parser.types(), parser.global_vars()};
    checker.run();

    CodeGenerator codegen{source_filename, parser.functions(),
                          parser.global_vars(), build_mode};
    codegen.run();

    std::cout << parser;

    return 0;
}
