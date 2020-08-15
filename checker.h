#ifndef CHECKER_H
#define CHECKER_H
#include "ast.h"
#include <vector>
#include <unordered_map>

class Checker {
private:
    const std::vector<Function>& m_functions;
    const std::vector<Type>& m_types;
    const std::unordered_map<std::string, NameType>& m_names_table;

    void validate_names();
public:
    Checker(const std::vector<Function>&, const std::vector<Type>&,
            const std::unordered_map<std::string, NameType>& names_table);
    void run();
};
#endif
