cmake_minimum_required(VERSION 3.17.0)
set(CMAKE_CXX_STANDARD 17)

project(libiq)

add_subdirectory(pybind11)
add_subdirectory(libsigmf)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/root/.pyenv/versions/3.9-dev")
set(Python3_DIR "/root/.pyenv/versions/3.9-dev/lib/cmake/python3.9")

set(Python_INCLUDE_DIRS "/root/.pyenv/versions/3.9-dev/include/python3.9")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

pybind11_add_module(libiq main.cpp src/converter.cpp)

target_include_directories(libiq PRIVATE ${Python_INCLUDE_DIRS})
include_directories(libiq PRIVATE pybind11/include)

set_target_properties(libiq PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(libiq PRIVATE VERSION_INFO="0.0.1")

target_link_libraries(libiq PRIVATE libsigmf Python3::Python)