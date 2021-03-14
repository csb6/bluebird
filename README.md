# Bluebird

Bluebird is an imperative programming language modeled after C++ and Ada.

The goal is to create a language that supports generic programming with
very strong typing.

It is still in the early stages, but right now it has a lexer, a parser,
a semantic analyzer, a code generator, and an optimizer. All stages of the compiler are still
a work-in-progress. At the moment, the compiler has only been built on macOS, but
it should work on any platform that LLVM supports.

## Currently Implemented Features

- Functions, variables, constants, and assignment
- Integer, boolean, character, reference, and array types
- Initializer lists and assignment for arrays
- Module-global variables
- If, else-if, and else statements
- While loops
- Recursion (with tail-call elimination in optimized builds)
- Type definitions for integer types, with ranges specified (no runtime range checks yet)
- Boolean type and an 8-bit character type
- Logical, comparison, bitwise, and arithmetic operators, as well as parentheses for grouping
- Detailed error checking, including typechecking and checks to ensure functions that
  return will return no matter the code path taken
- Out-of-order declarations of type and function names, preventing the need for forward declarations
- Debugging support (uses the DWARF format, which should work with at least gdb and lldb)
- Several code optimization passes

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

## Syntax

The syntax is very Ada-like, favoring English keywords over symbols and full words over abbreviations (where sensible). Here is a sample program:

```
type Positive is range 1 thru 500;

function fizzbuzz() is
    for n in Positive do
        if n mod 15 = 0 do
            print("fizzbuzz");
        else if n mod 3 = 0 do
            print("fizz");
        else if n mod 5 = 0 do
            print("buzz")
        else
            print_num(n);
        end if;
    end for;
end fizzbuzz; // end labels are optional
```

## Semantics

Parameters are passed by value. I plan on eventually implementing pointer types,
but for now there are only integer types and a single character type.

Variables are mutable by default; they can be made constant (i.e. allowing no
reassignment or modification) by adding the `constant` keyword to the declaration:

```
let age: constant Age := 99;
```

Three default types, `Integer` (a 32-bit integer), `Character` (an 8-bit character), and
`Boolean` do not have to be defined.

An arbitrary number of additional integer types can be defined, each of which is incompatible
with all others. To define an integer type, specify the desired range. The compiler
will try to choose a reasonable bit size for values of the new type:

```
type Dalmation_Count is range 1 thru 101;
// Or, equivalently:
type Dalmation_Count is range 1 upto 102;

let original_amt: Dalmation_Count := 96;
let puppy_amt: Dalmation_Count := 5;
let new_amt: Dalmation_Count := original_amt + puppy_amt; // Allowed
let pupp_amt2: Integer := 5;
// Compilation error (below): cannot use Dalmation_Count and Integer together
let new_amt2: Dalmation_Count := original_amt + puppy_amt2;
```

## Building

### Dependencies

- C++17 compiler
- CMake (>= 3.16.4)
- LLVM 10
- All other dependencies will be bundled into the `third_party/` directory

### Compiling the compiler

The build system is CMake.

To build, enter the `build/` directory of this project, then run:

```
cmake ..
cmake --build .
```

All build files/artifacts will be placed inside `build/`.

To rebuild from scratch run (from inside `build/`):

```
cmake --build . --clean-first
```

To build using multiple cores (which should be faster), you can also add
the `--parallel` flag like so:

```
cmake --build . --parallel
```

If you have any issues building, please leave a Github issue.

### Running the compiler

The compiler executable, `bluebird`, should be found in the `build` directory
after the build finishes. Pass it a filename to compile something
(e.g. `bluebird ../examples/arithmetic.bird`). An object file with the same name (but
a `.o` extension instead of a `.bird` extension) as well as an `a.out` executable
should be produced.

If you encounter a compiler bug, crash, or miscompilation, please leave a Github issue.

#### Compiler options

Compiler options are always placed first, before the source file name.
(e.g. `bluebird --debug ../examples/arithmetic.bird`)

Note that debug and optimization modes are mutually exclusive;
the last given flag will override any prior debug/optimization flags.

- `(no options given)`: Build with no optimizations or debug symbols
- `-g` or `--debug`: Build with debug symbols, no optimizations
- `-O` or `--optimize`: Build with optimizations, no debug symbols

## License

This program is licensed under the AGPLv3 license. The text of the AGPL can be found in
the LICENSE file.

Dependencies in the `third_party/` directory may have separate licenses; refer to their
documentation/source code for more information.
