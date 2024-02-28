#set the minimum version for cmake
cmake_minimum_required(VERSION 3.0)

#name of the project
project(libiq)

#set the C++ standard, in pybind11 is 11
set(CMAKE_CXX_STANDARD 11)

#include pybind11 in my configuration
add_subdirectory(pybind11)
#include SigMF in my configuration
add_subdirectory(SigMF)

#pybind11 function to crate Python modules extended in C++, 
#first parameter is the name of the Python module
#second one is the source code of the Python module
pybind11_add_module(libiq main.cpp)

#The add_library command in CMake is used to create a library from a set of source files. More than one can be specified.
#In this case, it's creating a static library named 'calculations' from the source file 'src/calculations.cpp'.
#A static library is a library that is linked at compile-time, and it's included as part of the final executable.
add_library(calculations STATIC src/calculations.cpp)

#This means that the 'calculations' target will be compiled with Position Independent Code (PIC).
#PIC is used for shared libraries, which can be loaded at different memory addresses in different runs without needing to be relinked.
#This increases the security and flexibility of the shared library.
set_target_properties(calculations PROPERTIES POSITION_INDEPENDENT_CODE ON)

#It's defining 'VERSION_INFO' as "0.0.1" for the 'libiq' target.
#Compile definitions are similar to '#define' directives in C++. They are replaced by their corresponding value at the start of the compilation process.
#So, whenever 'VERSION_INFO' appears in the source code for 'libiq', it will be replaced with "0.0.1".
target_compile_definitions(libiq PRIVATE VERSION_INFO="0.0.1")

#It's linking the 'calculations' library to the 'libiq' target.
#This means that when 'libiq' is compiled, the compiler will include code from the 'calculations' library.
#The 'PRIVATE' keyword in this context means that the 'calculations' library is linked only to 'libiq', and will not be linked to any other targets that might depend on 'libiq'.
target_link_libraries(libiq PRIVATE calculations libsigmf)
