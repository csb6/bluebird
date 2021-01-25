#include "codegenerator.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wdeprecated"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>
#include <iostream>
#include <lld/Common/Driver.h>
#pragma GCC diagnostic pop

// Contains parts of code generator that deal with generating and
// linking object files; these are essential to the code generator
// but not changed as often as the AST-walking code, so it is better
// to keep it in a separate compilation unit to make incremental builds
// more effective

void CodeGenerator::setup_llvm_targets()
{
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    const std::string target_triple{llvm::sys::getDefaultTargetTriple()};

    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(target_triple, error);
    if(!error.empty()) {
        std::cerr << "Codegen error: " << error << "\n";
        exit(1);
    }
    const llvm::TargetOptions options;

    m_target_machine = target->createTargetMachine(
        target_triple,
        "generic",
        "",
        options, llvm::Reloc::PIC_, {}, {});

    m_module.setDataLayout(m_target_machine->createDataLayout());
    m_module.setTargetTriple(target_triple);
}

void CodeGenerator::emit(const std::filesystem::path& object_file)
{
    std::error_code file_error;
    llvm::raw_fd_ostream output{object_file.string(), file_error};
    if(file_error) {
        std::cerr << "Codegen error: " << file_error.message() << "\n";
        return;
    }

    // We need this for some reason? Not really sure how to get around using it
    llvm::legacy::PassManager pass_manager;
    // TODO: add optimization passes
    m_target_machine->addPassesToEmitFile(pass_manager, output, nullptr,
                                          llvm::CodeGenFileType::CGFT_ObjectFile);
    pass_manager.run(m_module);
    output.flush();
}

void CodeGenerator::link(const std::filesystem::path& object_file,
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
