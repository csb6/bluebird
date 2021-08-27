#ifndef BLUEBIRD_CONTEXT_H
#define BLUEBIRD_CONTEXT_H
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
#include "magnum.h"

struct Context {
    std::vector<Magnum::Pointer<struct Function>> functions;
    std::vector<Magnum::Pointer<struct Type>> types;
    std::vector<Magnum::Pointer<struct Initialization>> global_vars;
    std::vector<Magnum::Pointer<struct IndexedVariable>> index_vars;
};
#endif // BLUEBIRD_CONTEXT_H
