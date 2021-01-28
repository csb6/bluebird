# Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
#    Copyright (C) 2021  Cole Blakley
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
target_sources(bluebird PRIVATE
  main.cpp
  lexer.cpp
  parser.cpp
  ast.cpp
  token.cpp
  checker.cpp
  codegenerator.cpp
  objectgenerator.cpp
  optimizer.cpp
  multiprecision.cpp
)
