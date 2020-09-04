#ifndef MULTIPRECISION_CPP_INT_H
#define MULTIPRECISION_CPP_INT_H
#include <string>
#include <iosfwd>

// Wrapper class for boost::multiprecision::cpp_int
class multi_int {
 public:
    explicit multi_int(const std::string&);
    multi_int(const multi_int&);
    ~multi_int();
    unsigned short bits_needed() const;
    std::string str() const;

    multi_int& operator+=(const multi_int&);
    multi_int& operator-=(const multi_int&);
    multi_int& operator*=(const multi_int&);
    multi_int& operator/=(const multi_int&);

    friend multi_int operator-(const multi_int&);
    friend multi_int operator+(const multi_int&, const multi_int&);
    friend multi_int operator-(const multi_int&, const multi_int&);
    friend multi_int operator*(const multi_int&, const multi_int&);
    friend multi_int operator/(const multi_int&, const multi_int&);
    friend bool operator==(const multi_int&, const multi_int&);
    friend bool operator!=(const multi_int&, const multi_int&);
    friend bool operator< (const multi_int&, const multi_int&);
    friend bool operator> (const multi_int&, const multi_int&);
    friend bool operator<=(const multi_int&, const multi_int&);
    friend bool operator>=(const multi_int&, const multi_int&);
    friend std::ostream& operator<<(std::ostream&, const multi_int&);
 private:
    struct multi_int_impl* impl;
};
#endif
