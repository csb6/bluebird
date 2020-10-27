# Bluebird

Bluebird is a work-in-progess imperative programming language modeled after C++ and Ada.

The goal is to create a language that supports generic programming with strong typing and syntax somewhere between Ada's
verbose but unambiguous syntax and C++'s more concise syntax.

It is still in the early stages, but right now it has a lexer and a parser implementing many basic features. Code generation and semantic analysis are in development, but aren't currently usable.

## Goals

- Ahead-of-time compiled language with native LLVM backend
- Ada-like syntax, but with less verbosity and little to no need for forward declarations
- Monomorphized generics with C++20-like concepts for constraining types
- First-class types
- Modules with mandatory specifications (a.k.a. header files), but no textual inclusion
- Very strong typing in the tradition of Ada, with few to no implicit conversions, but
with support for casting when needed
- Ranges as a core language construct, with support for user-defined ranges
- Types that can be constrained to a range
- Support for low-level bitwise operations and precise control over data representation
- Support for a large number of optional compile and run-time checks
- Built-in support for the partial evaluation of functions
- Support for compile-time evaluation of expressions and at least some subset of functions
- Standard library containing useful and efficient generic data structures and algorithms
- Highly modular compiler implemented with minimal dependencies and a reasonable build time
- Minimally complex build system for the compiler and have a simple, standard build
system for the language itself
- Easy way to create C and C++ bindings
- Compiler tools for interacting with each stage of the compiler, potentially with a GUI

## Currently Implemented Features

- Void functions
- Variables and constants
- If statements
- Type definitions for integer types
- Logical, comparison, bitwise, and arithmetic operators, as well as parentheses for grouping
- Detailed error checking in the lexing and parsing stages
- Automatic resolution of variable and function names, preventing the need for forward declarations within a file

## Syntax

The syntax is very Ada-like, favoring English keywords over symbols and full words over abbreviations (where sensible). Here is a sample
program demonstrating most of the language keywords:

```
// Comment
type Locker_Number is range 0 thru 6_000;
type Student_Id_Number is range 0 upto 12_000;

function bar is
   let a : Locker_Number = 3 + 4;
end bar;

/*
 Multi-line
 comment
*/
function main is
  let current_locker : Locker_Number = 0;
  // let locker2 : Locker_Number = -1; // Will not compile; out-of-range
  let foo : constant Locker_Number = 7 + 90 + 89 * (67 - 90 + 87);

  if (current_locker > 5 or current_locker == 0) do
     let id : Student_Id_Number = 45 | ~78;
  end if

end main;
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
(e.g. `bluebird examples/arithmetic.bird`). Right now, code generation is not complete, so no
binary files will be generated; however, you should see a printout of the abstract syntax tree
along with some generated LLVM IR instructions.

#### Other build modes

- `make debug`: build with debug symbols
- `make release`: build with optimizations

## License

This program is licensed under the AGPLv3 license. The text of the AGPL can be found in
the LICENSE file.

Dependencies in the `third_party/` directory may have separate licenses; refer to their
documentation/source code for more information.
