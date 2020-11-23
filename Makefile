.POSIX:
# Files that depend on LLVM
llvm_linked_files := main.o codegenerator.o
# All other files
object_files := lexer.o parser.o token.o ast.o checker.o multiprecision.o
c_files = third_party/mini-gmp/mini-gmp.o

exe_name := bluebird
# Need some version of clang++/clang as the compilers
# Point to path of clang++/clang version you want to use
compiler := clang++
c_compiler := clang
llvm_linker_flags := `llvm-config --ldflags --system-libs --libs all`
llvm_compiler_flags := `llvm-config --cxxflags`
flags := -std=c++17 -Wall -Wextra -pedantic-errors -Ithird_party
c_flags := -std=c99 -Ithird_party

# Link (default rule)
$(exe_name): $(object_files) $(llvm_linked_files) $(c_files)
	$(compiler) -o $(exe_name) $(llvm_linker_flags) $(flags) $(object_files) $(llvm_linked_files) $(c_files)

debug: flags += --debug
debug: $(exe_name)

release: flags += -O3 -flto -fno-rtti -ffunction-sections -fdata-sections
release: $(exe_name)

# Autogenerate header dependencies
-include $(object_files:.o=.d)
-include $(llvm_linked_files:.o=.d)

# Build
$(llvm_linked_files): %.o: %.cpp
	$(compiler) -c -MD -MF$*.d $(llvm_compiler_flags) $(flags) $*.cpp -o $*.o

$(object_files): %.o: %.cpp
	$(compiler) -c -MD -MF$*.d $(flags) $*.cpp -o $*.o

$(c_files): %.o: %.c
	$(c_compiler) -c -MD -MF$*.d $(c_flags) $*.c -o $*.o

clean:
	-rm $(exe_name)
	-rm *.o *.d
