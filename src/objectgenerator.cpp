/* Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
    Copyright (C) 2020-2022  Cole Blakley

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
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Host.h>
#pragma GCC diagnostic pop
#include <cstdlib>
#include "objectgenerator.h"
#include "error.h"

ObjectGenerator::ObjectGenerator(const char* linker_exe_path, llvm::Module& module)
    : m_object_file{module.getSourceFileName()}, m_module(module)
{
    if(linker_exe_path == nullptr) {
#ifdef __APPLE__
        m_linker_exe_path = "ld";
#elif defined _WIN32
        m_linker_exe_path = "link";
#elif defined __linux__
        m_linker_exe_path = "ld";
#else
        m_linker_exe_path = "";
#endif
    } else {
        m_linker_exe_path = linker_exe_path;
    }

    m_object_file.replace_extension(".o");
    m_object_file = m_object_file.filename(); // Place in current working directory
    std::error_code error_status;
    const auto file_status{std::filesystem::status(m_object_file, error_status)};

    if(error_status && file_status.type() != std::filesystem::file_type::not_found) {
        Error().put("When generating object file:").quote(m_object_file.c_str())
            .put("encountered filesystem error").quote(error_status.message())
            .raise();
    }


    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    const std::string target_triple{llvm::sys::getDefaultTargetTriple()};

    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple, error);
    if(!error.empty()) {
        Error().put("Codegen error:").quote(error).raise();
    }
    const llvm::TargetOptions options;

    m_target_machine = target->createTargetMachine(
        target_triple, "generic", "",
        options, llvm::Reloc::PIC_, {}, {});

    m_module.setDataLayout(m_target_machine->createDataLayout());
    m_module.setTargetTriple(target_triple);
}

void ObjectGenerator::emit()
{
    std::error_code file_error;
    llvm::raw_fd_ostream output{m_object_file.string(), file_error};
    if(file_error) {
        Error().put("During LLVM emit process:").quote(file_error.message()).raise();
    }

    // We need this for some reason? Not really sure how to get around using it
    llvm::legacy::PassManager pass_manager;
    m_target_machine->addPassesToEmitFile(pass_manager, output, nullptr,
                                          llvm::CodeGenFileType::CGFT_ObjectFile);
    pass_manager.run(m_module);
    output.flush();
}

void ObjectGenerator::link(std::filesystem::path&& exe)
{
    std::string command{m_linker_exe_path};
    command.reserve(command.size() + 20);

#ifdef __APPLE__
    command += " -L$(xcode-select --print-path)/SDKs/MacOSX.sdk/usr/lib -lSystem -o ";
#elif defined _WIN32
    command += " /WX /nologo ";
#elif defined __linux__
    command += " -o ";
#else
    Error().put("Note: linking not implemented for this platform, so"
                " no executable will be produced. Manually use linker to turn emitted"
                " object file into an executable\n");
    return;
#endif

    command += exe.string() + " " + m_object_file.string();

#ifdef __linux__
    command += " -lc";
#endif

    int error_code = system(command.c_str());
    if(error_code != 0) {
        Error().put("Linker invocation:").quote(command).newline()
            .put("Failed with error code: ").put(error_code).raise();
    }
}
