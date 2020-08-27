# Files that depend on LLVM
llvm_linked_files := main.o codegenerator.o
# All other files
object_files := lexer.o parser.o token.o ast.o checker.o

exe_name := bluebird
compiler := clang++
llvm_linker_flags := `llvm-config --ldflags --system-libs --libs all`
llvm_compiler_flags := `llvm-config --cxxflags`
flags := -std=c++17 -Wall -Wextra -pedantic-errors -Ithird_party

# Link (default rule)
$(exe_name): $(object_files) $(llvm_linked_files)
	$(compiler) -o $(exe_name) $(llvm_linker_flags) $(flags) $(object_files) $(llvm_linked_files)

debug: flags += --debug
debug: link

release: flags += -O3 -flto -fno-rtti -ffunction-sections -fdata-sections
release: link

# Autogenerate header dependencies
-include $(object_files:.o=.d)
-include $(llvm_linked_files:.o=.d)

# Build
$(llvm_linked_files): %.o: %.cpp
	$(compiler) -c $(llvm_compiler_flags) $(flags) $*.cpp -o $*.o
	$(compiler) -MM $(llvm_compiler_flags) $(flags) $*.cpp > $*.d

$(object_files): %.o: %.cpp
	$(compiler) -c $(flags) $*.cpp -o $*.o
	$(compiler) -MM $(flags) $*.cpp > $*.d

clean:
	-rm $(exe_name)
	-rm *.o *.d
