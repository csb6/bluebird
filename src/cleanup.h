#ifndef BLUEBIRD_CLEANUP_H
#define BLUEBIRD_CLEANUP_H
/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2021  Cole Blakley

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
#include <vector>
#include <unordered_map>
#include "magnum.h"

/* This file contains a class that does some cleanup operations between the
   parsing and checking stages. This includes constant folding/type resolution
   of literal values. The AST will be modified during this pass, folding each
   binary/unary operation involving only literal operations into individual literals.
   Additionally, after this pass, all literals (including the ones creating via
   constant folding) will have a type assigned to them that is not the default
   IntLiteral, BoolLiteral, etc. type. The type will be determined based on context.
*/

class Cleanup {
    std::vector<Magnum::Pointer<struct Function>>& m_functions;
    std::vector<Magnum::Pointer<struct Initialization>>& m_global_vars;
    std::unordered_map<const struct Type*, Magnum::Pointer<struct PtrType>> m_anon_ptr_types;
public:
    Cleanup(std::vector<Magnum::Pointer<Function>>&,
            std::vector<Magnum::Pointer<Initialization>>& global_vars);
    void run();
};
#endif
