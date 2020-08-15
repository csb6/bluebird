#include "checker.h"
#include <iostream>

Checker::Checker(const std::vector<Function>& functions,
                 const std::vector<Type>& types,
                 const std::unordered_map<std::string, NameType>& names_table)
    : m_functions(functions), m_types(types), m_names_table(names_table)
{}

void Checker::validate_names()
{
    // Check that all functions and types in this module have a definition
    // TODO: Have mechanism where types/functions in other modules are resolved
    for(const auto&[name, name_type] : m_names_table) {
        switch(name_type) {
        case NameType::DeclaredFunct:
            std::cerr << "Error: Function `" << name << "` is declared but has no body\n";
            exit(1);
        case NameType::DeclaredType:
            std::cerr << "Error: Type `" << name << "` is declared but has no body\n";
            exit(1);
        default:
            // All other kinds of names are acceptable
            break;
        }
    }
}

void Checker::run()
{
    validate_names();
}
