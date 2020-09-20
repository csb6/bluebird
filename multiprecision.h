#ifndef MULTIPRECISION_CPP_INT_H
#define MULTIPRECISION_CPP_INT_H
#include <string>
#include <iosfwd>

// Wrapper class for boost::multiprecision::cpp_int
class multi_int {
 public:
    multi_int();
    explicit multi_int(const std::string&);
    // Copy Constructors
    multi_int(const multi_int&);
    multi_int& operator=(const multi_int&);
    // Move Constructors
    multi_int(multi_int&&) = default;
    multi_int& operator=(multi_int&&) = default;
    ~multi_int();

    auto bits_needed() const { return m_bits_needed; };
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
    void set_bits_needed();

    struct multi_int_impl* impl;
    unsigned short m_bits_needed = 1;
};
#endif
