#include "multiprecision.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>

struct multi_int_impl {
    multi_int_impl() = default;
    explicit multi_int_impl(const std::string& v) : value(v) {}
    boost::multiprecision::cpp_int value;
};

multi_int::multi_int() : impl(new multi_int_impl())
{}

multi_int::multi_int(const std::string& value) : impl(new multi_int_impl(value))
{
    set_bits_needed();
}

multi_int::multi_int(const multi_int& other)
{
    const multi_int_impl& oi = *other.impl;
    impl = new multi_int_impl{oi};
    m_bits_needed = other.m_bits_needed;
}

multi_int& multi_int::operator=(const multi_int& other)
{
    const multi_int_impl& oi = *other.impl;
    impl = new multi_int_impl{oi};
    m_bits_needed = other.m_bits_needed;
    return *this;
}

multi_int::~multi_int()
{
    
    delete impl;
}

void multi_int::set_bits_needed()
{
    if(impl->value < 0) {
        impl->value = abs(impl->value) * 2;
    }

    unsigned short exp = 0;
    boost::multiprecision::cpp_int curr_power{1};
    while(curr_power < impl->value) {
        curr_power *= 2;
        ++exp;
    }
    m_bits_needed = exp + 1;
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
