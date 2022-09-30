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
  cleanup.cpp
  codegenerator.cpp
  objectgenerator.cpp
  optimizer.cpp
  multiprecision.cpp
  error.cpp
  constanteval.cpp
  ast.h
  checker.h
  cleanup.h
  codegenerator.h
  error.h
  lexer.h
  magnum.h
  multiprecision.h
  objectgenerator.h
  optimizer.h
  parser.h
  token.h
  visitor.h
  constanteval.h
  context.h
  util.h
)

add_subdirectory(bluebird-ir)
