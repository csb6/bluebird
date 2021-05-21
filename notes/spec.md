// To generate, run: pandoc -f markdown+grid_tables -t html spec.md > spec.html

# The Bluebird Language

## Goals

- **Correctness** - the language features should help you write code that is correct, meaning
it does what your design says it should do. Designs can be expressed using types and
concepts that enforce invariants, checking for inconsistencies in your implementation.
Additionally, the compiler itself, which will be implemented to fulfill this specification
document, should be correct to a reasonable degree of certainty.

- **Performance** - the language should be able to be compiled to efficient machine code
on par with current C++ or Ada compilers. The language should allow the programmer to
specify data layout in detail, as well as having options to tweak and change data layout
without affecting the usage of the data structures. Additionally, complex constructs such
as ranges should compile down to efficient machine code.

- **Genericity** - the language should have extensive support for generic subroutines, which
can be restricted using concepts and instantiated to code that is as fast as hand-written
non-generic code. The standard library should be written to be as generic as possible,
with the ability to reuse algorithms on a variety of data structures, including custom
user-made structures that implement the proper concepts.

- **Consistency** - the language should have standard ways to express common operations
within programs instead of inventing ad-hoc ways to do these things. For instance, all
iteration should be expressed using ranges, all data structures should be expressed using
types, and all functions should be amenable to chaining into a sequence of operations.

- **Compile-time execution** - the language should support selective execution of code at
compile time, such as enabling certain classes of functions to be evaluated before
compile-time. Additionally, there should be support for partial evaluation, meaning
that programs can be specialized for certain inputs, making them more efficient at
runtime. This should all mesh well with the system of generics.

- **Composability** - the language should allow operations to be composed and passed around
similar to how data can be composed and passed around. In particular, ranges should be
lazily-evaluated, enabling complex filtering operations to be built-up and used.
Support for coroutines/generators/first-class functions should make composing operations
simple to accomplish.

## Syntax

The grammar is defined using a modified BNF notation. Operator precedence is not part of
the BNF grammar, instead being represented in separate tables containing numeric values
used to disambiguate expressions containing operators.

Text contained within `/` and `/` is a regular expression.

```
<program> ::= <stmt_list>


<identifier> ::= / [a-zA-Z]+[a-zA-Z0-9_]* /
<type_label> ::= <identifier> | constant <identifier> | ref <identifier>
               | ptr <identifier> | [<identifier>] of <identifier>
<var_decl> ::= <identifier> : <type_label>


<function_header> ::= function <identifier>(<param_list>)
                    | function <identifier>(<param_list>): <type_label>
<param_list> ::= <identifier> : <type_label>, <param_list>
               | <identifier> : <type_label>
<function_def> ::= <function_header> is <stmt_list> end <identifier>;
                 | <function_header> is <stmt_list> end;
<type_decl> ::= type <identifier> is range <expr> <range_op> <expr>;
              | type <identifier> is new <identifier>;
              | subtype <identifier> is <identifier>;
              | type <identifier> is ref <identifier>;
              | type <identifier> is ptr <identifier>;
              | type <identifier> is [<identifier>] of <identifier>;
              | type <identifier> is record <field_list> end record;
<field_list> ::= <var_decl>; <field_list> | <var_decl>;


<stmt_list> ::= <stmt> <stmt_list> | <stmt> | <empty>
<stmt> ::= <function_def> | <type_decl> | <basic_stmt> | <init_stmt> | <assign_stmt>
         | <if_stmt> | <loop_stmt> | <while_stmt> | <for_stmt> | <return_stmt>
         | <yield_stmt>
<basic_stmt> ::= <expr>;
<init_stmt> ::= let <var_decl>;
              | let <var_decl> := <expr>;
              | let <identifier> := <expr>;
<assign_stmt> ::= <var_expr> <assign_symbol> <expr>;
<if_stmt> ::= if <expr> do <stmt_list> end;
            | if <expr> do <stmt_list> end if;
            | if <expr> do <stmt_list> else <if_stmt>
            | if <expr> do <stmt_list> else <stmt_list> end;
            | if <expr> do <stmt_list> else <stmt_list> end if;
<loop_stmt> ::= loop <stmt_list> end;
              | loop <stmt_list> end loop;
<while_stmt> ::= while <expr> <loop_stmt>
<for_stmt> ::= for <identifier> in <expr> <loop_stmt>
<return_stmt> ::= return; | return <expr>;
<yield_stmt> ::= yield; | yield <expr>;


<expr> ::= <literal> | <var_expr> | <unary_op_expr> | <bin_op_expr> | ( <expr> )
         | <init_list> | <index_op>
<literal> := / \-?[0-9]+ / | "/ [char]* /" | '<char>' | '/ \\[abfnrtv\\\'\"] /'
           | true | false |  / \-?[0-9]+\.[0-9]+ /
<var_expr> ::= <identifier> | <dot_op_expr> | <index_op>
<init_list> ::= { <expr_list> }
<expr_list> ::= <expr>, <expr_list> | <expr>
<funct_call> ::= <identifier>(<expr_list>)


<index_op> ::= <var_expr>[<expr>]
<unary_op> ::= <see section Operator Precedence/Unary operators>
<bin_op> ::= <see section Operator Precedence/Binary operators>
<range_op> ::= <see section Operator Precedence/Binary operators>
<dot_op> ::= <see section Operator Precedence/Binary operators>
<unary_op_expr> ::= <unary_op> <expr>
<dot_op_expr> ::= <expr> <dot_op> <expr>
<bin_op_expr> ::= <expr> <bin_op> <expr> | <dot_op_expr>
<assign_symbol> ::= <see section Assignment symbols>
```

### Assignment symbols

Used to associate new values with a variable, potentially performing an extra operation
before doing so.

+--------+-----------+
|Symbol  |Description|
+:=======+:==========+
|:=      |Basic      |
|        |assignment |
+--------+-----------+
|+=      |*a += b*   |
|        |becomes *a |
|        |:= a + b*  |
+--------+-----------+
|-=      |*a -= b*   |
|        |becomes *a |
|        |:= a - b*  |
+--------+-----------+
|/=      |*a /= b*   |
|        |becomes *a |
|        |:= a / b*  |
+--------+-----------+
|\*=     |*a \*= b*  |
|        |becomes *a |
|        |:= a \* b* |
+--------+-----------+
|%=      |*a %= b*   |
|        |becomes *a |
|        |:= a mod b*|
+--------+-----------+
|&=      |*a &= b*   |
|        |becomes *a |
|        |:= a and b*|
+--------+-----------+
||=      |*a |= b*   |
|        |becomes *a |
|        |:= a or b* |
+--------+-----------+
|^=      |*a ^= b*   |
|        |becomes *a |
|        |:= a xor b*|
+--------+-----------+
|<<=     |*a <<= b*  |
|        |becomes *a |
|        |:= a << b* |
+--------+-----------+
|\>>=    |*a >>= b*  |
|        |becomes *a |
|        |:= a >> b* |
+--------+-----------+


### Operator precedence

Higher precedence means evaluated first. Note that the assignment symbols
(e.g. `:=`, `+=`) are not operators and so cannot be used within expressions.

#### Unary operators

All are right associative.

*Unary arithmetic operator*: {\-}

*Unary bitwise operator*: {not}

*Unary logical operator*: {not}

*Unary pointer operator*: {val}

+--------+----------+------------+
|Operator|Precedence|Description  |
+:=======+:=========+:===========+
|\-      |9         |Flips sign  |
|        |          |of numeric  |
|        |          |type        |
+--------+----------+------------+
|not     |9         |Flips       |
|        |          |boolean     |
|        |          |value       |
+--------+----------+------------+
|val     |9         |Dereferences|
|        |          |a ptr type  |
|        |          |            |
+--------+----------+------------+

#### Binary operators

All are left associative.

*Binary arithmetic operators*: {/, \*, mod, \+, \-}

*Binary bitwise operators*: {<<, \<<, and, xor, or}

*Binary logical operators*: {and, xor, or}

*Comparison operators*: {<, >, <=, >=, =, !=}

*Equality operators*: {=, !=}

*Range operators*: {thru, upto}

+--------+----------+---------------+
|Operator|Precedence|Description    |
+:=======+:=========+:==============+
|.       |10        |Function       |
|        |          |call/field     |
|        |          |access         |
+--------+----------+---------------+
|as      |8         |Type conversion|
+--------+----------+---------------+
|/       |7         |Division       |
+--------+----------+---------------+
|\*      |7         |Multiplication |
+--------+----------+---------------+
|mod     |7         |Modulus        |
+--------+----------+---------------+
|\+      |6         |Addition       |
+--------+----------+---------------+
|\-      |6         |Subtraction    |
+--------+----------+---------------+
|<<      |6         |Shift left     |
+--------+----------+---------------+
|\>>     |6         |Shift right    |
+--------+----------+---------------+
|and     |5         |Logical/bitwise|
|        |          |AND            |
+--------+----------+---------------+
|xor     |4         |Logical/bitwise|
|        |          |XOR            |
+--------+----------+---------------+
|or      |3         |Logical/bitwise|
|        |          |OR             |
+--------+----------+---------------+
|thru    |2         |Inclusive range|
+--------+----------+---------------+
|upto    |2         |Exclusive range|
+--------+----------+---------------+
|=       |1         |Equal to       |
+--------+----------+---------------+
|!=      |1         |Not equal to   |
+--------+----------+---------------+
|<       |1         |Less than      |
+--------+----------+---------------+
|\>      |1         |Greater than   |
+--------+----------+---------------+
|<=      |1         |Less than or   |
|        |          |equal to       |
+--------+----------+---------------+
|\>=     |1         |Greater than or|
|        |          |equal to       |
+--------+----------+---------------+


## Semantics

### Expressions

Expressions are groups of operand and operators that define a computation. There are many
kind of expressions:

- *Literals*
   - *String literals*
   - *Character literals*
   - *Integer literals*
   - *Boolean literals*
   - *Float literals*
- *Variable expressions*
- *Unary expressions*
- *Binary expressions*
- *Function calls*
- *Index operations*
- *Initialization lists*

Expressions must be either part of another expression or part of a statement. Expressions cannot
be standalone.

Expressions, at execution time (whether runtime or compile-time execution) are eagerly evaluated
into values. These values are then used in the evaluation of other expressions and/or statements.

### Statements

Statements are standalone parts of a program, and all are terminated in a semicolon. There are
several kinds of statements:

- *Basic statements* - consist of a single expression
- *Initialization statements* - create a variable and optionally assign the result of an expression
to it
- *Assignment statements* - assign the result of an expression to a variable
- *Block statements* - create a new scope that contains other statements
- *If statements* (includes various forms: if-else, if-else-if, if-else-if-else, etc.)
- *Loop statements* - plain loops with no header conditions
- *While statements*
- *For statements*
- *Return statements*
- *Yield statements*
- *Function definitions*
- *Type definitions*

Statements can contain expressions (like *Basic statements*) and/or other statements
(like *Block statements* or *Function definitions*).

Statements, at execution time (whether runtime or compile-time execution) are eagerly evaluated.
The side-effects of statements affect the execution of the program and the subsequent evaluation of
other statements/expressions. Statements themselves do not evaluate into a value; however, they can
return a value to a caller (e.g. *Return statements*) or set the value of a variable.

### Scope

Scopes are regions of the program in which various statements, functions, types, and variables
exist. Scopes are defined by blocks, meaning that entering a new block means entering a new scope.

A scope `b` is a descendent of scope `a` if scope `b` is defined within scope `a` or within
a descendent of scope `a`. Entities defined within a scope are only visible within that scope
and its descendent scopes.

There are several ways to open new blocks:

1. *Block statements*
2. *Bodies of if statements, all kinds of loops*
3. *Function definitions*
4. *Bodies of record type definitions*
5. *The module-global scope* - this is the implicit scope of a translation unit. All entities
defined outside of any other explicit block (e.g. top-level functions) are in this scope. All
other scopes in a translation unit are descendents of this scope.

### Variables

Variables are named entities in the program that can hold values, that is, the results of executing
expressions and/or statements.

There are two kinds of variables:

- *Mutable variables* - can be assigned values an arbitrary number of times
- *Constants* - must be assigned an initial value, then cannot be reassigned

Variables can be used in the evaluation of expressions through *variable expressions*, which
are expressions consisting solely of a variable name. There are two different possible usage
semantics for variables:

- *Value usage semantics* - the result of evaluating the variable expression is the value stored
at the variable. This is the behavior when the variable is not a ref type.
- *Reference usage semantics* - the result of evaluating the variable expression is the value
obtained by dereferencing the variable. This is the behavior of ref types in most contexts
(except when a variable with a ref type is being assigned to another variable with a ref type).

### Types

Every value has a type, as does every variable. When initializing or assigning a
variable, the expression will be evaluated to a value; this value must have the
same type as the variable it is being assigned to.

The type of a variable, once specified in an initialization or declaration, cannot be changed
during execution or by a subsequent statement/expression.

*Primitive types* are builtin types that do not need to be defined explicitly and are available
in all programs. They are: `Integer`, `Boolean`, `Character`, and `Float`.

Primitive types have a set of *primitive operations*, which are the builtin operators that can
be used on these types. These operators are inherited when new types are derived from primitive
types.

*Anonymous types* are types that are created as part of a statement that is not a type definition
statement. The places where this can occur is in initialization statements, the declaration
of fields in record types, or in the formal parameter lists of functions. These types are structurally
compatible with other types (e.g. a `ref Integer` parameter can accept any argument with a type that
has a definition reading `ref Integer` or a descendent subtype of any such type).

*Ref types* are types representing aliases of variable of another type. Variables with this type
cannot be stored in records or arrays, and cannot be returned from functions. They function much like
pointer types, except no explicit dereferences are needed and their usage is restricted.

*Pointer types* are types that hold a memory address, typically that of another variable. The
value at that address can be accessed by dereferencing a variable with this type using the `val`
unary operator.

### Assignment/initialization

`a` is the the type of a variable `Va`, `b` is the type of
an expression `Eb` being assigned to `Va`.

This rule applies to assignment/initialization statements, argument passing into a
function call, and the assignment of values from an initialization list to the corresponding
fields in an array or record.

If one of the following conditions is true, `Eb` is said to be *assignable* to `Va`:

- `a` and `b` are the same type
- `b` is a descendent of `a`
- `b` is a literal type that can be implicitly converted to `a`. This is valid if one of
the following is true:

   - `a` is `Character`, a subtype of `Character`, or a type whose ancestor is `Character`
      and `b` is a character literal that meets the constraints of `a`.
   - `a` is a ranged integer type and `b` is an integer literal that meets the constraints
     of `a`.
   - `a` is a floating point type and `b` is a floating point or integer literal that
     meets the constraints of `a`.
   - `a` is an array type containing type `c`. `c` is either: `Character`, a subtype of
     `Character`, or a type whose ancestor is `Character`. `Eb` is an initialization list
     containing expressions which are assignable to variables of type `c` and the length
     of `Eb` is less than or equal to the length of `a`, if `a` has a specified length.
   - `a` is a record type. `Eb` is an initialization list containing `n` expressions
     which are each assignable to the corresponding fields of `a`. `n` = the number of
     fields in `a`.

Otherwise, the usage is illegal and the program is invalid.

If an initialization statement does not include an initial expression (e.g. `let a : Integer;`),
then the initial value of the variable is undefined.

### Unary expressions

Unary expressions have form: `op` `a`. There are two kinds of unary expressions: primitive
and user-defined.

Primitive unary operators are builtin and operate on primitive types.

User-provided binary operators map to functions that have the same name as the operator
(enclosed in double-quotes) and one parameter. These overloads are found either in the standard
library, third-party libraries, or the current file.

#### Primitive unary operators

`a` is the type of an expression `Ea`, `op` is a binary operator. There is an expression
`op Ea`.

`a` can only be used with `op` if one of the following rules is matched and true:

- If `a` is a numeric type, then `op` is an *unary arithmetic operator* or *unary bitwise operator*.
- If `a` is a boolean type, then `op` is an *unary logical operator*.

If `a` is not a primitive type and/or does not match these conditions, the program is invalid.

#### User-provided unary operator overloads

`a` is the type of an expression `Ea`, `op` is a binary operator. There is an expression `op Ea`.
There are two different search criteria (see next section for search order) to find
the right overloads:

1. A function with the signature: `function "op"(a)` is searched for; if found, then
`op Ea` is translated into a call to that overload with `Ea` as its argument.

2. An overload is searched for with signature: `function "op"(c)` such that `c` is
a descendent of `a`. If more than one such overload is found, the program is incorrectly formed.
If only one match is found, this overload is used similar to the first case (see prior paragraph).

#### Selection of operator overload

First, a function matching the first set of search criteria under the User-Provided Unary Operator
Overloads section should be looked for; in other words, try to find an overload exactly matching the
types of this unary expression's operand.

If not successful, search for a primitive operator using the procedure outlined under the Primitive
Unary Operators section.

If still not successful, search for a user-provided overload using the second set of search criteria
under the User-Provided Unary Operator Overloads section.

If still not successful, then the program is invalid; the operator is being used improperly.

### Binary expressions

Binary expressions have form: `a` `op` `b`. There are two kinds of binary expressions: primitive
and user-defined.

Primitive binary operators are builtin and operate on primitive types.

User-provided binary operators map to functions that have the same name as the operator
(enclosed in double-quotes) and two parameters. These overloads are found either in the standard
library, third-party libraries, or the current file.

#### Primitive binary operators

`a` is the type of an expression `Ea`, `op` is a binary operator, and `b` is the type of an
expression `Eb`. `a` and `b` are primitive types. There is a expression `Ea op Eb`.

`a` and `b` can only be legally used together in a primitive binary expression
if they are *compatible*; otherwise, the program is invalid.

For `a` and `b` to be *compatible* in a primitive binary expression, two conditions must hold:

1. Exactly one of the following must be true:
   - `a` and `b` are the same type
   - `b` is a descendent of `a`
   - `a` is a descendent of `b`

2. One of these rules must be matched and true:
   - If `a` is a numeric type, then `op` is a *binary arithmetic operator*, *binary bitwise operator*,
   *comparison operator*, or a *range operator*.
   - If `a` is a boolean type, then `op` is a *binary logical operator* or an *equality operator*
   - If `a` is a character type, then `op` is a *comparison operator*.

If `a` and `b` are not primitive types and/or do not match these conditions, the program is invalid.

#### User-provided binary operator overloads

`a` is the type of an expression `Ea`, `op` is a binary operator, and `b` is the type of an
expression `Eb`. There is an expression `Ea op Eb`. There are two different search criteria
(see next section for search order) to find the right overloads:

1. A function with the signature: `function "op"(a, b)` is searched for; if found, then
`Ea op Eb` is translated into a call to that overload with `Ea` and `Eb` as its arguments.

2. An overload is searched for with signature: `function "op"(c, d)` such that `c` is
equal to `a` or a descendent of `a`, and `d` is equal to `b` or a descendent of `b`. If more
than one such overload is found, the program is incorrectly formed. If only one match is found,
this overload is used similar to the first case (see prior paragraph).

#### Selection of operator overload

First, a function matching the first set of search criteria under the User-Provided Binary Operator
Overloads section should be looked for; in other words, try to find an overload exactly matching the
two types of this binary expression's operands.

If not successful, search for a primitive operator using the procedure outlined under the Primitive
Binary Operators section.

If still not successful, search for a user-provided overload using the second set of search criteria
under the User-Provided Binary Operator Overloads section.

If still not successful, then the program is invalid; the operator is being used improperly.