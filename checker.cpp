#include "checker.h"
#include <iostream>

Checker::Checker(const std::vector<Function>& functions,
                 const std::vector<Type>& types)
    : m_functions(functions), m_types(types)
{}

void Checker::run()
{
}
