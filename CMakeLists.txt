# Bluebird compiler - ahead-of-time compiler for the Bluebird language using LLVM.
#    Copyright (C) 2020-2021  Cole Blakley
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
cmake_minimum_required(VERSION 3.19.1)

project(Bluebird)

# For some reason CMake can't find LLVM on macOS,
# so we need to give it some help
find_package(LLVM 10.0 REQUIRED CONFIG
    HINTS /usr/local/opt/llvm/lib/cmake/llvm)

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wdeprecated)
endif()

add_executable(bluebird)
include_directories(src third_party third_party/mini-gmp)
add_subdirectory(src)
add_subdirectory(third_party/mini-gmp)

llvm_map_components_to_libnames(LLVM_LIBS core native option passes)
list(APPEND LLVM_LIBS lldDriver lldCommon lldCore lldYAML lldReaderWriter)
target_link_libraries(bluebird ${LLVM_LIBS})

if(CMAKE_SYSTEM MATCHES "Darwin")
    # Without this, won't link proper std::filesystem code
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15)
endif()

#libfuzzer build mode - fuzzes lexer (does not work with AppleClang)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message("Note: fuzzer target enabled")
    add_executable(fuzzer src/fuzzer.cpp src/lexer.cpp src/token.cpp src/error.cpp)
    target_compile_options(fuzzer PRIVATE -fsanitize=fuzzer)
    target_link_libraries(fuzzer PRIVATE -fsanitize=fuzzer)
    # Set -DFUZZER_MODE (C++ macro definition)
    target_compile_definitions(fuzzer PRIVATE FUZZER_MODE)
else()
    message("Note: fuzzer target not enabled (compiler is not Clang)")
endif()
