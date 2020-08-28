# Bluebird

Bluebird is a work-in-progess imperative programming language modelled after C++ and Ada.

The goal is to create a language that supports generic programming with strong typing and syntax somewhere between Ada's
verbose but unambiguous syntax and C++'s more concise syntax.

It is still in the early stages, but right now it has a lexer and a parser that implement a lot of the basic features. Code generation and
semantic analysis are starting to develop, but aren't done yet.

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

  if (current_locker > 5 or current_locker) do
     let id : Student_Id_Number = 45 | ~78;
  end if

end main;
```

## Building

### Dependencies

- A C++17 compiler
- LLVM 10
- All other dependencies will be bundled in the `third_party/` directory

### Build Process

There is a `Makefile`, which contains a few variables for setting up the build process
(e.g. setting the path to your C++ compiler of choice).

Simply run `make` to build the compiler. You should end up with an executable named
`bluebird` in the working directory. Pass in a filename to compile it
(e.g. `bluebird parser-test.txt`). Right now, code generation is not complete, so no
binary files will be generated, but you should see a printout of the parse tree.

#### Other build modes

- `make debug`: build with debug mode
- `make release` build with optimizations