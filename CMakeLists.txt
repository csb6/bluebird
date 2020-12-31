cmake_minimum_required(VERSION 3.16.4)

project(Bluebird)

# For some reason CMake can't find LLVM on macOS,
# so we need to give it some help
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    set(LLVM_DIR /usr/local/opt/llvm/lib/cmake/llvm)
endif()
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
# To find the lld .a files
link_directories(${LLVM_LIBRARY_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wdeprecated)
endif()

add_executable(bluebird
  main.cpp
  lexer.cpp
  parser.cpp
  ast.cpp
  token.cpp
  checker.cpp
  codegenerator.cpp
  multiprecision.cpp)
include_directories(third_party third_party/mini-gmp)
add_subdirectory(third_party/mini-gmp)

llvm_map_components_to_libnames(llvm_libs core native option)
target_link_libraries(bluebird ${llvm_libs}
                      lldDriver lldCommon lldCore lldYAML lldReaderWriter lldMachO)
