# LIBIQ-010: A Library for I/Q Sample Analysis

## Introduction
`libiq` is library designed to facilitate the management and analysis of I/Q samples. It provides a suite of tools and functionalities that allow users to handle I/Q samples.

## Dependencies
`libiq` is built on a foundation of several libraries:

- **libsigmf**: A library that provides standardized format for storing signal metadata in signal capture files.
- **matio**: A library for reading and writing Matlab MAT files.
- **SWIG (Simplified Wrapper and Interface Generator)**: A software development tool that connects programs written in C and C++ with a variety of high-level programming languages. In `libiq`, we used SWIG to create bindings between the C++ code and Python, although SWIG supports many other languages.
- **FFTW (Fastest Fourier Transform in the West)**: FFTW is a C subroutine library for computing the discrete Fourier transform (DFT) in one or more dimensions, of arbitrary input size, and of both real and complex data.

## Building the Project
Building `libiq` is a straightforward process. Here are the steps you need to follow:

1. First be sure to build and install all the dependencies, in particular:
    - **matio**: it can be build following the instructions on [matio's Github page](https://github.com/tbeu/matio?tab=readme-ov-file#22-building-matio)
    - **libsigmf**: it is an header only library but you need to build its dependencies following the instructions on [libsigmf's Github page](https://github.com/deepsig/libsigmf)
    - **SWIG**: it can be build following the instructions on [SWIG's Github page](https://github.com/swig/swig)
    - **FFTW**: it can be build following the instructions on [FFTW's Home Page](https://www.fftw.org/)
3. Run `./remove_build.sh` to clean up the build directory. This removes all the files from the previous build, ensuring a fresh start.
4. Execute `./build_auto.sh` to initiate the build process. This script compiles the source code and creates the necessary modules.
5. Finally, run `./execute_prova.sh` to execute the `prova.py` script located in the `examples/prova.py` directory. This allows you to test the functionality.

## Features
`libiq` comes with a set of classes:

- **Converter**: this class allows you to convert the I/Q sample's file in different format. 
    - `from_bin_to_mat`: This function allows you to convert a `.bin` file containing an I/Q sample into a `.mat` file. This is particularly useful when you need to analyze I/Q samples using tools that support `.mat` files.
    - `from_mat_to_sigmf`: This function enables you to convert a `.mat` file into a `.sgmf` file. This is beneficial when you want to store signal metadata along with the signal data in a standardized format.
- **Analyzer**: this class provides a set of functions for analyzing I/Q samples.
    - `fast_fourier_transform`: This function takes the path to an I/Q sample file as input and returns the Fast Fourier Transform (FFT) of the I/Q samples. The FFT is a mathematical technique that transforms a function of time into a function of frequency.
    - `calculate_PSD`: This function calculates the Power Spectral Density (PSD) of the I/Q samples. The PSD is a measure of the power intensity in the frequency domain. The function takes the path to an I/Q sample file and the sample rate as inputs.
    - `generate_IQ_Scatterplot`: This function takes the path to an I/Q sample file as input and generates a scatter plot of the I/Q samples. The scatter plot is a graphical representation of the variation of a signal.
    - `generate_IQ_Spectrogram`: This function takes in input the path to an I/Q sample file, the overlap and the window size and generates a spectrogram of the I/Q samples. A spectrogram is a visual representation of the spectrum of frequencies in a signal as it varies with time. The colors in the spectrogram represent the intensity of the signal at that particular frequency and at that particular moment in time. Warmer colors (like red) indicate a high signal intensity, while cooler colors (like blue) indicate a low signal intensity.
    - `get_iq_sample`: This function reads the file at the given `input_file_path` and returns the IQ sample represented as a complex vector.
    - `complex_part_iq_sample`: This function reads the file at the given `input_file_path` and returns the complex part of the IQ data sample.
    - `real_part_iq_sample`: This function reads the file at the given `input_file_path` and returns the real part of the IQ data sample.



## Examples
```
import libiq
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

```
import libiq
import src_python.scatterplot as scplt
import src_python.spectrogram as sp

#input_file_path is the path to .bin or .iq file
analyzer = libiq.Analyzer()
iq = analyzer.get_iq_sample(input_file_path)
complex_iq = analyzer.complex_part_iq_sample(input_file_path)
real_iq = analyzer.real_part_iq_sample(input_file_path)

fft_result = analyzer.fast_fourier_transform(input_file_path)

sample_rate = 1000;
psd_result = analyzer.calculate_PSD(input_file_path, sample_rate)

iq = analyzer.get_iq_sample(input_file_path)
scplt.scatterplot(iq)

psd = analyzer.generate_IQ_Spectrogram(input_file_path, 0, 256)
sample_rate = 20 * 5
sp.spectrogram(psd, sample_rate)
```
