#ifndef UTIL_H
#define UTIL_H
/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2022  Cole Blakley

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

#define BLUEBIRD_MOVEABLE(type) \
    type(const type&) = default; \
    type& operator=(const type&) = default;

#define BLUEBIRD_COPYABLE(type) \
    type(type&&) = default; \
    type& operator=(type&&) = default;
#endif // UTIL_H