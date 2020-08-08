#!/usr/bin/env sh

set -e

compiler=clang++
flags="-std=c++17 -Wall -Wextra -pedantic-errors"
include_paths="third_party"
exe_name=compiler

$compiler $flags $@ -I$include_paths -o $exe_name main.cpp lexer.cpp parser.cpp token.cpp ast.cpp
