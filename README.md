# Bluebird

Bluebird is a work-in-progess imperative programming language modeled after C++ and Ada.

The goal is to create a language that supports generic programming with strong typing and syntax somewhere between Ada's
verbose but unambiguous syntax and C++'s more concise syntax.

It is still in the early stages, but right now it has a lexer and a parser that implement a lot of the basic features. Code generation and
semantic analysis are starting to develop, but aren't done yet.

## Goals

- Ahead-of-time compiled language with native LLVM backend
- Ada-like syntax, but with less verbosity and little to no need for forward declarations
- Monomorphized generics with C++20-like concepts for constraining types
- First-class types
- Modules with mandatory specifications (a.k.a. header files), but no textual inclusion
- Very strong typing in the tradition of Ada, with few to no implicit conversions, but
with support for casting when needed
- Ranges as a core language construct, with support for users-defined ranges
- Types that support constraining values to a range
- Support for low-level bitwise operations and ways to precisely control the representation of data
- Support for a large number of optional compile and run-time checks
- Built-in support for partial evaluation of functions
- Support for simple compile-time evaluation of expressions and calls to at least some
functions
- Standard library consisting of a large amount of generic data structures and algorithms
- Compiler implemented with minimal dependencies/complexity/build times
- Use a minimally complex build system for the compiler and have a simple, standard build
solution for the language itself
- Easy way to create C and C++ bindings

## Currently Implemented Features

- Void functions
- Variables and constants
- If statements
- Type definitions for integer types
- Logical, comparison, bitwise, and arithmetic operators, as well as parentheses for grouping
- Detailed error checking throughout the lexer and parser
- Automatic resolution of variable and function names, preventing need for forward declarations within a file

## Syntax

The syntax is very Ada-like, favoring English keywords over symbols and full words over abbreviations (where sensible). Here is a sample
program demonstrating most of the language keywords and what the compiler can handle at
this early stage:

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
(e.g. `bluebird parser-test.txt`). Right now, code generation is not complete, so no
binary files will be generated; however, you should see a printout of the parse tree.

#### Other build modes

- `make debug`: build with debug symbols
- `make release`: build with optimizations