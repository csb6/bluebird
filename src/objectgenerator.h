#ifndef OBJECT_GENERATOR_H
#define OBJECT_GENERATOR_H
/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2021  Cole Blakley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include <filesystem>

namespace llvm {
    class Module;
    class TargetMachine;
};

class ObjectGenerator {
public:
    ObjectGenerator(const char* linker_exe_path, llvm::Module&);
    // Emit the object file for this module first, then link it into an executable
    void emit();
    void link(std::filesystem::path&& exe);
private:
    std::filesystem::path m_object_file;
    const char* m_linker_exe_path;
    llvm::Module& m_module;
    llvm::TargetMachine* m_target_machine;
};
#endif
