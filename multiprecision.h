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

    multi_int& operator+=(const multi_int&);
    multi_int& operator-=(const multi_int&);
    multi_int& operator*=(const multi_int&);
    multi_int& operator/=(const multi_int&);

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
    friend std::ostream& operator<<(std::ostream&, const multi_int&);
 private:
    mpz_t m_number;
};
#endif
