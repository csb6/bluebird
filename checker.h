#ifndef CHECKER_H
#define CHECKER_H
#include <vector>

class Checker {
private:
    const std::vector<struct Function*>& m_functions;
    const std::vector<struct RangeType*>& m_types;
    void check_types(const struct Statement*) const;
    void check_types(const Statement*, const struct Expression*) const;
public:
    Checker(const std::vector<Function*>&, const std::vector<struct RangeType*>&);
    void run();
};
#endif
