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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/raw_ostream.h>
#pragma GCC diagnostic pop
#include <cstdlib>
#include "objectgenerator.h"
#include "error.h"

void emit(llvm::Module& module, llvm::TargetMachine* target_machine,
          const std::filesystem::path& object_file)
{
    std::error_code file_error;
    llvm::raw_fd_ostream output{object_file.string(), file_error};
    if(file_error) {
        Error().put("During LLVM emit process:").quote(file_error.message()).raise();
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
    int error_code;

#ifdef __APPLE__
    error_code = system(("ld -sdk_version 10.14 -o " + exe_file.string()
                         + " " + object_file.string()
                         + " -lSystem").c_str());
#elif defined _WIN32
    error_code = system(("link /WX /nologo " + exe_file.string()
                         + " " + object_file.string()).c_str());
#elif defined __linux__
    error_code = system(("ld -o" + exe_file.string()
                         + " " + object_file.string() + " -lc").c_str());
#else
    error_code = 0;
    Error().put("Note: linking not implemented for this platform, so"
                " no executable will be produced. Manually use linker to turn emitted"
                " object file into an executable\n");
#endif

    if(error_code != 0) {
        Error().put("Linker failed with error code:").quote(error_code).raise();
    }
}
