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
#include "objectgenerator.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetMachine.h>
#include <iostream>
#include <lld/Common/Driver.h>
#pragma GCC diagnostic pop

void emit(llvm::Module& module, llvm::TargetMachine* target_machine,
          const std::filesystem::path& object_file)
{
    std::error_code file_error;
    llvm::raw_fd_ostream output{object_file.string(), file_error};
    if(file_error) {
        std::cerr << "Codegen error: " << file_error.message() << "\n";
        return;
    }

    // We need this for some reason? Not really sure how to get around using it
    llvm::legacy::PassManager pass_manager;
    target_machine->addPassesToEmitFile(pass_manager, output, nullptr,
                                        llvm::CodeGenFileType::CGFT_ObjectFile);
    pass_manager.run(module);
    output.flush();
}

void link(const std::filesystem::path& object_file,
          const std::filesystem::path& exe_file)
{
#ifdef __APPLE__
    const char* args[] = { "lld", "-sdk_version", "10.14", "-o", exe_file.c_str(),
                           object_file.c_str(), "-lSystem" };
    if(!lld::mach_o::link(args, false, llvm::outs(), llvm::errs())) {
        std::cerr << "Linker failed\n";
        exit(1);
    }
#elif defined _WIN32
    const char* args[] = { "lld", "-o", exe_file.c_str(), object_file.c_str() };
    if(!lld::coff::link(args, false, llvm::outs(), llvm::errs())) {
        std::cerr << "Linker failed\n";
        exit(1);
    }
#elif defined __linux__
    const char* args[] = { "lld", "-o", exe_file.c_str(), object_file.c_str() };
    if(!lld::elf::link(args, false, llvm::outs(), llvm::errs())) {
        std::cerr << "Linker failed\n";
        exit(1);
    }
#else
    std::cerr << "Note: linking not implemented for this platform, so"
        " no executable will be produced. Manually use linker to turn emitted"
        " object file into an executable\n";
#endif
}