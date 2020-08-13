#!/usr/bin/env sh

set -e

compiler=clang++
llvm_flags="`llvm-config --cxxflags --ldflags --system-libs --libs all`"
#`flags` will override `llvm_flags` if they conflict
flags="-std=c++17 -Wall -Wextra -pedantic-errors"
include_paths="third_party"
exe_name=compiler

$compiler $llvm_flags $flags $@ -I$include_paths -o $exe_name main.cpp lexer.cpp parser.cpp token.cpp ast.cpp codegenerator.cpp
