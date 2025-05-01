# LibIQ Library

## Dependencies
`libiq` is built on a foundation of several libraries:

- **libsigmf**: A library that provides standardized format for storing signal metadata in signal capture files.
- **matio**: A library for reading and writing Matlab MAT files.
- **SWIG (Simplified Wrapper and Interface Generator)**: A software development tool that connects programs written in C and C++ with a variety of high-level programming languages. In `libiq`, we used SWIG to create bindings between the C++ code and Python, although SWIG supports many other languages.
- **FFTW (Fastest Fourier Transform in the West)**: FFTW is a C subroutine library for computing the discrete Fourier transform (DFT) in one or more dimensions, of arbitrary input size, and of both real and complex data.

## Building the Project
Building `libiq` is a straightforward process. Here are the steps you need to follow:

Be sure to build and install all the dependencies, in particular:
    - **matio**: it can be build following the instructions on [matio's Github page](https://github.com/tbeu/matio?tab=readme-ov-file#22-building-matio)
    - **libsigmf**: it is an header only library but you need to build its dependencies following the instructions on [libsigmf's Github page](https://github.com/deepsig/libsigmf)
    - **SWIG**: it can be build following the instructions on [SWIG's Github page](https://github.com/swig/swig)
    - **FFTW**: it can be build following the instructions on [FFTW's Home Page](https://www.fftw.org/)

A script that performs the installation of all the dependencies is given toghether with the library, to execute is you need to run 
```
./libiq_installer.sh
```
This script will update all submodules, build and install all the necessary libraries, creates a python virtual environment and download all python dependencies inside this virtual environment.

To activate the virtual environment run:
```
source .libiqvenv310/bin/activate
```

Once this is done, to build LibIQ, you need to run:
```
hatch env create
```
Once this is done, run:
```
hatch build
```
Once this is done, run:
```
pip install dist/libiq-0.1.0.tar.gz
```

