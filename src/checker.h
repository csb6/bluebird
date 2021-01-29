/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2021  Cole Blakley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef CHECKER_H
#define CHECKER_H
#include <vector>
#include <CorradePointer.h>

namespace Magnum = Corrade::Containers;

/* This file contains a class that performs semantic analysis on
   a given AST, including typechecking, type resolution for
   various kinds of universal literals, and checks to ensure that
   returning functions return on all code paths. The AST should remain mostly
   unchanged after being run through this stage (besides the Literal nodes).
*/

class Checker {
private:
    std::vector<Magnum::Pointer<struct Function>>& m_functions;
    std::vector<Magnum::Pointer<struct RangeType>>& m_types;
    std::vector<Magnum::Pointer<struct Initialization>>& m_global_vars;

    // Data about the current things being typechecked/analyzed
    struct BBFunction* m_curr_funct = nullptr;

    friend struct ReturnStatement;
public:
    Checker(std::vector<Magnum::Pointer<Function>>&,
            std::vector<Magnum::Pointer<RangeType>>&,
            std::vector<Magnum::Pointer<Initialization>>& global_vars);
    /* Analyze the full contents of all functions, types, and global variables
       of a module */
    void run();
};
#endif
