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
#ifndef BLUEBIRD_ERROR_H
#define BLUEBIRD_ERROR_H

#ifdef FUZZER_MODE
/* ret_val is the return type of the containing function (don't pass an argument
   if returning void) */
  #define fatal_error(ret_val) return ret_val
#else
  #include <cstdlib>
  #define fatal_error(ret_val) (exit(1))
#endif
#endif
