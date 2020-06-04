#!/usr/bin/env sh

set -e

compiler=clang++
flags="-std=c++17 -Wall -Wextra -pedantic-errors"
exe_name=compiler

$compiler $flags $@ -o $exe_name main.cpp lexer.cpp parser.cpp token.cpp
