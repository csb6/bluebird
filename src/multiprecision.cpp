/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2022  Cole Blakley

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
#include "multiprecision.h"
#include <iostream>
#include <cstdlib>

multi_int::multi_int()
{
    mpz_init(m_number);
}

multi_int::multi_int(std::string_view value)
{
    mpz_init_set_str(m_number, value.data(), 10);
}

multi_int::multi_int(const multi_int& other)
{
    mpz_init_set(m_number, other.m_number);
}

multi_int& multi_int::operator=(const multi_int& other)
{
    mpz_init_set(m_number, other.m_number);
    return *this;
}

multi_int::multi_int(multi_int&& other)
{
    m_number[0] = std::move(other.m_number[0]);
    other.m_number[0] = {};
}

multi_int& multi_int::operator=(multi_int&& other)
{
    m_number[0] = std::move(other.m_number[0]);
    if(&other != this) {
        other.m_number[0] = {};
    }
    return *this;
}

multi_int::~multi_int()
{   
    mpz_clear(m_number);
}

std::string multi_int::str() const
{
    char* str = nullptr;
    str = mpz_get_str(str, 10, m_number);
    std::string result{str};
    free(str);
    return result;
}

size_t multi_int::bits_needed() const
{
    return mpz_sizeinbase(m_number, 2);
}

multi_int& multi_int::operator+=(unsigned int other)
{
    mpz_add_ui(m_number, m_number, other);
    return *this;
}

multi_int& multi_int::operator+=(const multi_int& other)
{
    mpz_add(m_number, m_number, other.m_number);
    return *this;
}

multi_int& multi_int::operator-=(const multi_int& other)
{
    mpz_sub(m_number, m_number, other.m_number);
    return *this;
}

multi_int& multi_int::operator-=(unsigned int other)
{
    mpz_sub_ui(m_number, m_number, other);
    return *this;
}

multi_int& multi_int::operator*=(unsigned int other)
{
    mpz_mul_ui(m_number, m_number, other);
    return *this;
}

multi_int& multi_int::operator*=(const multi_int& other)
{
    mpz_mul(m_number, m_number, other.m_number);
    return *this;
}

multi_int& multi_int::operator/=(const multi_int& other)
{
    // Divide, truncate result
    mpz_tdiv_q(m_number, m_number, other.m_number);
    return *this;
}

multi_int& multi_int::mod(const multi_int& other)
{
    mpz_mod(m_number, m_number, other.m_number);
    return *this;
}

multi_int& multi_int::rem(const multi_int& other)
{
    mpz_tdiv_r(m_number, m_number, other.m_number);
    return *this;
}

multi_int& multi_int::negate()
{
    mpz_neg(m_number, m_number);
    return *this;
}

multi_int& multi_int::ones_complement()
{
    mpz_com(m_number, m_number);
    return *this;
}

bool multi_int::is_negative() const
{
    return mpz_sgn(m_number) < 0;
}

multi_int& multi_int::operator&= (const multi_int& other)
{
    mpz_and(m_number, m_number, other.m_number);
    return *this;
}

multi_int& multi_int::operator|=(const multi_int& other)
{
    mpz_ior(m_number, m_number, other.m_number);
    return *this;
}

multi_int& multi_int::operator^=(const multi_int& other)
{
    mpz_xor(m_number, m_number, other.m_number);
    return *this;
}

multi_int operator-(const multi_int& a)
{
    multi_int result{a};
    mpz_neg(result.m_number, a.m_number);
    return result;
}

bool operator==(const multi_int& a, const multi_int& b)
{
    return mpz_cmp(a.m_number, b.m_number) == 0;
}

bool operator!=(const multi_int& a, const multi_int& b)
{
    return mpz_cmp(a.m_number, b.m_number) != 0;
}

bool operator<(const multi_int& a, const multi_int& b)
{
    return mpz_cmp(a.m_number, b.m_number) < 0;
}

bool operator>(const multi_int& a, const multi_int& b)
{
    return mpz_cmp(a.m_number, b.m_number) > 0;
}

bool operator<=(const multi_int& a, const multi_int& b)
{
    return a < b || a == b;
}

bool operator>=(const multi_int& a, const multi_int& b)
{
    return a > b || a == b;
}

unsigned long int to_int(multi_int&& n)
{
    return mpz_get_ui(n.m_number);
}

std::ostream& operator<<(std::ostream& output, const multi_int& n)
{
    char* str = nullptr;
    str = mpz_get_str(str, 10, n.m_number);
    output << str;
    free(str);
    return output;
}
