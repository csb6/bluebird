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
cmake_minimum_required(VERSION 3.16.4)

project(Bluebird)

if(CMAKE_SYSTEM MATCHES "Darwin")
    # For some reason CMake can't find LLVM on macOS,
    # so we need to give it some help
    find_package(LLVM 10.0 REQUIRED CONFIG
        HINTS /usr/local/opt/llvm/lib/cmake/llvm)
else()
    find_package(LLVM 10.0 REQUIRED CONFIG)
endif()
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

llvm_map_components_to_libnames(LLVM_LIBS core native option passes)
list(APPEND LLVM_LIBS lldDriver lldCommon lldCore lldYAML lldReaderWriter)

#Try to link only parts of lld that generate this platform's executables
if(CMAKE_SYSTEM MATCHES "Darwin")
    list(APPEND LLVM_LIBS lldMachO)
    # Without this, won't link proper std::filesystem code
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15)
elseif(CMAKE_SYSTEM MATCHES "Windows")
    list(APPEND LLVM_LIBS WindowsManifest lldCOFF)
elseif(CMAKE_SYSTEM MATCHES "Linux")
    list(APPEND LLVM_LIBS lldELF)
endif()

target_link_libraries(bluebird ${LLVM_LIBS})
