/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020  Cole Blakley

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

class Checker {
private:
    const std::vector<struct Function*>& m_functions;
    const std::vector<struct RangeType*>& m_types;

    // For typechecking, virtual functions attached to each Expression subclass.
    // These functions are defined in checker.cpp
    friend struct StringLiteral;
    friend struct CharLiteral;
    friend struct IntLiteral;
    friend struct FloatLiteral;
    friend struct LValueExpression;
    friend struct UnaryExpression;
    friend struct BinaryExpression;
    friend struct FunctionCall;

    void check_types(const struct Statement*) const;
public:
    Checker(const std::vector<Function*>&, const std::vector<struct RangeType*>&);
    void run() const;
};
#endif
