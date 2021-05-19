// To generate, run: pandoc -f markdown+grid_tables -t html spec.md > spec.html

# The Bluebird Language

## Goals

- **Correctness** - the language features should help you write code that is correct, meaning
it does what your design says it should do. Designs can be expressed using types and
concepts that enforce invariants, checking for inconsistencies in your implementation

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
               | ptr <identifier> | [<identifier>] of <identifer>


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


<stmt_list> ::= <stmt> <stmt_list> | <stmt> | <empty>
<stmt> ::= <function_def> | <type_decl> | <basic_stmt> | <init_stmt> | <assign_stmt>
         | <if_stmt> | <loop_stmt> | <while_stmt> | <for_stmt> | <return_stmt>
         | <yield_stmt>
<basic_stmt> ::= <expr>;
<init_stmt> ::= let <identifier> : <type_label>;
              | let <identifier> : <type_label> := <expr>;
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


<range_op> ::= upto | thru
<unary_op_expr> ::= <unary_op> <expr>
<dot_op_expr> ::= <expr> . <expr>
<bin_op_expr> ::= <expr> <bin_op> <expr> | <dot_op_expr>
<index_op> ::= <var_expr>[<expr>]
<unary_op> ::= <see section Operator Precedence/Unary operators>
<bin_op> ::= <see section Operator Precedence/Binary operators>
<assign_symbol> ::= <see section Assignment symbols>
```

### Assignment symbols

Used to associate new values with a variable, potentially evaluating an extra operation
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


### Operator Precedence

Higher precedence means evaluated first. Note that the assignment symbols
(e.g. `:=`, `+=`) are not operators and so cannot be used within expressions.

#### Unary operators

All are right associative.

+--------+----------+------------+
|Operator|Precedence|Decription  |
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

### Types

Every value has a type, as does every variable. When initializing or assigning a
variable, the expression will be evaluated to a value; this value must have the
same type as the variable it is being assigned to.

The type of a variable, once specified in an initialization or declaration, cannot be changed.

#### Legal Usage

Let there be two types, `a` and `b`, that are being used together. Types can be legally used
together in a couple of different contexts, which are:

a. *Assignment/Initialization*  - `a` is the the type of a variable `Va`, `b` is the type of
   an expression `Eb` being assigned to `Va`.

    This context includes assignment/initialization statements, argument passing into a
    function call, and the assignment of values from an initialization list to the corresponding fields in an array or record.

    If `Eb` fulfills one of the following conditions, `Eb` is said to be *assignable* to `Va`:
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

b. *Primitive Binary Operators* - `a` is the type of an expression `Ea`, `op` is a
binary operator, and `b` is the type of an expression `Eb`. There is an expression `Ea op Eb`.

    Note that user-provided binary operators/overloads map to function calls, so an overload
    matching `op(`Ea`, `Eb`)` would be searched for using the semantics of Context A.
    Context B only applies for primitive types, which are: integer, boolean, character,
    and floating point types. These types have predefined operations that we call
    "primitive".

    `a` and `b` can only be legally used together in a primitive binary expression
    if they are *compatible*; otherwise, the program is invalid and should be rejected.

    For `a` and `b` to be *compatible* in a primitive binary expression, exactly
    one of the following must be true:

    - `a` and `b` are the same type
    - `b` is a descendent of `a`
    - `a` is a descendent of `b`