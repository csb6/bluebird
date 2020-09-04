#include "multiprecision.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>

struct multi_int_impl {
    explicit multi_int_impl(const std::string& v) : value(v) {}
    boost::multiprecision::cpp_int value;
};


multi_int::multi_int(const std::string& value) : impl(new multi_int_impl(value))
{}

multi_int::multi_int(const multi_int& other)
    : impl{new multi_int_impl{*other.impl}}
{}

multi_int::~multi_int()
{
    delete impl;
}

unsigned long multi_int::bits_needed() const
{
    // First, try and fit into common int sizes
    for(unsigned long i = 0; i <= 12; ++i) {
        if(impl->value < (2ul << i)) {
            return i + 1;
        }
    }

    // Else, find a roughly appropriate size
    /*For x = 2^n, solve for n.
      n = log(x) / log(2)
      Ex:
      log(10^17) / log(2) = 56.472...
      so 10^17 roughly = 2^56
     */
    size_t precision = impl->value.str().size();
    unsigned long range = std::pow(10, precision);
    return (log2(range) + 1) / log2(2);
}

std::string multi_int::str() const
{
    return impl->value.str();
}

multi_int& multi_int::operator+=(const multi_int& other)
{
    impl->value += other.impl->value;
    return *this;
}

multi_int& multi_int::operator-=(const multi_int& other)
{
    impl->value -= other.impl->value;
    return *this;
}

multi_int& multi_int::operator*=(const multi_int& other)
{
    impl->value *= other.impl->value;
    return *this;
}

multi_int& multi_int::operator/=(const multi_int& other)
{
    impl->value /= other.impl->value;
    return *this;
}


multi_int operator-(const multi_int& a)
{
    multi_int result{a};
    result.impl->value *= -1;
    return result;
}

multi_int operator+(const multi_int& a, const multi_int& b)
{
    multi_int result{a};
    result.impl->value += b.impl->value;
    return result;
}

multi_int operator-(const multi_int& a, const multi_int& b)
{
    multi_int result{a};
    result.impl->value -= b.impl->value;
    return result;
}

multi_int operator*(const multi_int& a, const multi_int& b)
{
    multi_int result{a};
    result.impl->value *= b.impl->value;
    return result;
}

multi_int operator/(const multi_int& a, const multi_int& b)
{
    multi_int result{a};
    result.impl->value /= b.impl->value;
    return result;
}

bool operator==(const multi_int& a, const multi_int& b)
{
    return a.impl->value == b.impl->value;
}

bool operator!=(const multi_int& a, const multi_int& b)
{
    return a.impl->value != b.impl->value;
}

bool operator< (const multi_int& a, const multi_int& b)
{
    return a.impl->value < b.impl->value;
}

bool operator> (const multi_int& a, const multi_int& b)
{
    return a.impl->value > b.impl->value;
}

bool operator<=(const multi_int& a, const multi_int& b)
{
    return a.impl->value <= b.impl->value;
}

bool operator>=(const multi_int& a, const multi_int& b)
{
    return a.impl->value >= b.impl->value;
}

std::ostream& operator<<(std::ostream& output, const multi_int& n)
{
    return output << n.impl->value;
}
