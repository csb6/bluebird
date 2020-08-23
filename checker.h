#ifndef CHECKER_H
#define CHECKER_H
#include "ast.h"
#include <vector>

class Checker {
private:
    const std::vector<Function>& m_functions;
    const std::vector<Type>& m_types;
    void check_types(const Statement*) const;
    void check_types(const Statement*, const Expression*) const;
public:
    Checker(const std::vector<Function>&, const std::vector<Type>&);
    void run();
};
#endif
