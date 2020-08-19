#ifndef CHECKER_H
#define CHECKER_H
#include "ast.h"
#include <vector>

class Checker {
private:
    const std::vector<Function>& m_functions;
    const std::vector<Type>& m_types;
public:
    Checker(const std::vector<Function>&, const std::vector<Type>&);
    void run();
};
#endif
