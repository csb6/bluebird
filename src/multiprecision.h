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
#ifndef MULTIPRECISION_CPP_INT_H
#define MULTIPRECISION_CPP_INT_H
#include <string>
#include <iosfwd>
#include <mini-gmp/mini-gmp.h>

// Wrapper class for mini-gmp, a C multiprecison number library
class multi_int {
 public:
    multi_int();
    explicit multi_int(const std::string&);
    multi_int(const multi_int&);
    multi_int& operator=(const multi_int&);
    multi_int(multi_int&&) = default;
    multi_int& operator=(multi_int&&) = default;
    ~multi_int();

    size_t bits_needed() const;
    std::string str() const;

    multi_int& operator+=(unsigned int);
    multi_int& operator+=(const multi_int&);
    multi_int& operator-=(const multi_int&);
    multi_int& operator-=(unsigned int);
    multi_int& operator*=(const multi_int&);
    multi_int& operator*=(unsigned int);
    multi_int& operator/=(const multi_int&);
    multi_int& mod(const multi_int&);
    multi_int& rem(const multi_int&);
    multi_int& negate();
    multi_int& ones_complement();
    bool is_negative() const;
    multi_int& operator&= (const multi_int&);
    multi_int& operator|= (const multi_int&);
    multi_int& operator^= (const multi_int&);

    friend multi_int operator-(const multi_int&);
    /*friend multi_int operator+(const multi_int&, const multi_int&);
    friend multi_int operator-(const multi_int&, const multi_int&);
    friend multi_int operator*(const multi_int&, const multi_int&);
    friend multi_int operator/(const multi_int&, const multi_int&);*/
    friend bool operator==(const multi_int&, const multi_int&);
    friend bool operator!=(const multi_int&, const multi_int&);
    friend bool operator< (const multi_int&, const multi_int&);
    friend bool operator> (const multi_int&, const multi_int&);
    friend bool operator<=(const multi_int&, const multi_int&);
    friend bool operator>=(const multi_int&, const multi_int&);
    friend unsigned long int to_int(multi_int&&);
    friend std::ostream& operator<<(std::ostream&, const multi_int&);
 private:
    mpz_t m_number;
};
#endif
