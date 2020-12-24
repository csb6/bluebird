# Bluebird

Bluebird is a work-in-progess imperative programming language modeled after C++ and Ada.

The goal is to create a language that supports generic programming with
very strong typing.

It is still in the early stages, but right now it has a lexer, a parser, and
a semantic analyzer. Code generation currently generates to object files, and it
creates executables on macOS only (for now). All stages of the compiler are still
a work-in-progress.

## Goals

- Ahead-of-time compiled language with native LLVM backend
- Ada-like syntax, but with less verbosity and no need for forward declarations
when defining interdependent types or functions
- Monomorphized generics with C++20-like concepts for constraining types
- First-class types
- Modules with mandatory specifications (a.k.a. header files), but no textual inclusion
- Very strong typing in the tradition of Ada, with no implicit conversions, but
with support for casting when needed
- Ranges as a core language construct, with support for user-defined ranges
- Types that can be constrained to a range
- Support for low-level operations and precise control over data representation
- Support for a large number of compile-time and run-time checks
- Support for compile-time evaluation and partial evaluation of functions
- Standard library containing useful and efficient generic data structures and algorithms
- Modular compiler implemented with minimal dependencies and a reasonable build time
- Minimally complex build environment for the compiler and a simple, standard build
system for the language itself
- Easy way to create C and C++ bindings
- Compiler tools for interacting with each stage of the compiler, potentially with a GUI

## Currently Implemented Features

- Void functions
- Variables, constants, and assignment
- If, else-if, and else statements
- Type definitions for integer types, with ranges specified (no runtime range checks yet)
- Logical, comparison, bitwise, and arithmetic operators, as well as parentheses for grouping
- Detailed error checking in the lexing and parsing stages
- Automatic resolution of variable and function names, preventing the need for forward declarations within a file

## Syntax

The syntax is very Ada-like, favoring English keywords over symbols and full words over abbreviations (where sensible). Here is a sample program:

```
type Positive is range 1 thru 500;

function fizzbuzz() is
    for n in Positive do
        if n mod 15 == 0 do
            print("fizzbuzz");
        else if n mod 3 == 0 do
            print("fizz");
        else if n mod 5 == 0 do
            print("buzz")
        else
            print_num(n);
        end if;
    end for;
end fizzbuzz; // 'fizzbuzz' label at end is optional
```

## Building

### Dependencies

- C++17 compiler
- LLVM 10
- All other dependencies will be bundled into the `third_party/` directory

### Build Process

There is a `Makefile`, which contains a few variables for setting up the build process
(e.g. choosing your C++ compiler). Make any adjustments that you need in this file.

To build the compiler, simply run `make`. You should end up with an executable named
`bluebird` in the working directory. Pass it a filename to compile something
(e.g. `bluebird examples/arithmetic.bird`). An object file with the same name (but
a `.o` extension instead of a `.bird` extension) should be produced. On macOS, a
runnable `a.out` executable will also be generated.

#### Other build modes

- `make debug`: build with debug symbols
- `make release`: build with optimizations

## License

This program is licensed under the AGPLv3 license. The text of the AGPL can be found in
the LICENSE file.

Dependencies in the `third_party/` directory may have separate licenses; refer to their
documentation/source code for more information.
