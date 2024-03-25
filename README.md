# LIBIQ-010: A Library for I/Q Sample Analysis

## Introduction
`libiq` is library designed to facilitate the management and analysis of I/Q samples. It provides a suite of tools and functionalities that allow users to handle I/Q samples.

## Dependencies
`libiq` is built on a foundation of several libraries:

- **libsigmf**: A library that provides standardized format for storing signal metadata in signal capture files.
- **matio**: A library for reading and writing Matlab MAT files.
- **SWIG (Simplified Wrapper and Interface Generator)**: A software development tool that connects programs written in C and C++ with a variety of high-level programming languages. In `libiq`, we used SWIG to create bindings between the C++ code and Python, although SWIG supports many other languages.

## Building the Project
Building `libiq` is a straightforward process. Here are the steps you need to follow:

1. First be sure to build and install all the dependencies, in particular:
    - **matio**: it can be build following the instructions on [matio's Github page](https://github.com/tbeu/matio?tab=readme-ov-file#22-building-matio)
    - **libsigmf**: it is an header only library but you need to build its dependencies following the instructions on [libsigmf's Github page](https://github.com/deepsig/libsigmf)
    - **SWIG**: it can be build following the instructions on [SWIG's Github page](https://github.com/swig/swig)
3. Run `./remove_build.sh` to clean up the build directory. This removes all the files from the previous build, ensuring a fresh start.
4. Execute `./build_auto.sh` to initiate the build process. This script compiles the source code and creates the necessary modules.
5. Finally, run `./execute_prova.sh` to execute the `prova.py` script located in the `examples/prova.py` directory. This allows you to test the functionality.

## Features
`libiq` comes with a set of classes:

- **Converter**: this class allows you to convert the I/Q sample's file in different format. 
    - `from_bin_to_mat`: This function allows you to convert a `.bin` file containing an I/Q sample into a `.mat` file. This is particularly useful when you need to analyze I/Q samples using tools that support `.mat` files.
    - `from_mat_to_sigmf`: This function enables you to convert a `.mat` file into a `.sgmf` file. This is beneficial when you want to store signal metadata along with the signal data in a standardized format.

## Examples
```
converter = libiq.Converter() 

#insert metadata values according to Sigmf standard
converter.freq_lower_edge = 213456
converter.freq_upper_edge = 3456768
converter.sample_rate = 23456
converter.frequency = 567890
converter.global_index = 9999
converter.sample_start = 1
converter.hw = "hello"
converter.version = "1.0.0"

#input_file_path1 is the path to .bin or .iq file
#output_file_path1 is the path to .mat file
converter.from_bin_to_mat(input_file_path1, output_file_path1)

#input_file_path2 is the path to .mat file
#output_file_path2 is the path to .sigmf-meta file
converter.from_mat_to_sigmf(input_file_path2, output_file_path2)
```
