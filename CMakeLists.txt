cmake_minimum_required(VERSION 3.16.4)

project(Bluebird)

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    # For some reason CMake can't find LLVM on macOS,
    # so we need to give it some help
    set(LLVM_DIR /usr/local/opt/llvm/lib/cmake/llvm)
    # Without this, won't link proper std::filesystem code
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15)
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

add_executable(bluebird)
include_directories(src third_party third_party/mini-gmp)
add_subdirectory(src)
add_subdirectory(third_party/mini-gmp)

llvm_map_components_to_libnames(llvm_libs core native option)
target_link_libraries(bluebird ${llvm_libs}
                      lldDriver lldCommon lldCore lldYAML lldReaderWriter)

#Try to link parts of lld that generate this platform's executables
if(CMAKE_SYSTEM MATCHES "Darwin")
    target_link_libraries(bluebird lldMachO)
elseif(CMAKE_SYSTEM MATCHES "Windows")
    target_link_libraries(bluebird WindowsManifest lldCOFF)
elseif(CMAKE_SYSTEM MATCHES "Linux")
    target_link_libraries(bluebird lldELF)
endif()
