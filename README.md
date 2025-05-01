# LibIQ Library

##Overview
LibIQ is a modular and extensible library designed for the manipulation, visualization, and classification of I/Q (In-phase and Quadrature) samples. It provides a Python interface built on a C++ backend using SWIG, enabling high performance for signal analysis tasks while maintaining ease of use in Python environments.

## Dependencies
`LibIQ` relies on several external libraries for full functionality:

- **libsigmf**: A header-only C++ library for reading and writing metadata-compliant signal capture files using the Signal Metadata Format (SigMF). Note: it includes dependencies that must be manually built and installed.
- **matio**: A C library for reading and writing MATLAB .mat files, including support for version 7.3 files (via HDF5).
- **SWIG (Simplified Wrapper and Interface Generator)**: A tool that generates Python bindings for the C++ backend, allowing Python applications to call native C++ functions.
- **FFTW (Fastest Fourier Transform in the West)**: A C library for computing the discrete Fourier transform (DFT) efficiently for arbitrary input sizes.

## Build and Installation
Building `libiq` is a straightforward process. Here are the steps you need to follow:

Be sure to build and install all the dependencies, in particular:
    - **matio**: it can be build following the instructions on [matio's Github page](https://github.com/tbeu/matio?tab=readme-ov-file#22-building-matio)
    - **libsigmf**: it is an header only library but you need to build its dependencies following the instructions on [libsigmf's Github page](https://github.com/deepsig/libsigmf)
    - **SWIG**: it can be build following the instructions on [SWIG's Github page](https://github.com/swig/swig)
    - **FFTW**: it can be build following the instructions on [FFTW's Home Page](https://www.fftw.org/)
    - **Python 3.10**: Required to run the Python interface and associated scripts. Installed and isolated using venv.

LibIQ provides a fully automated installer script that takes care of:
    - Cloning the repository (if needed)
    - Switching to the appropriate Git branch
    - Initializing all submodules
    - Building and installing all C/C++ dependencies
    - Creating a Python virtual environment (.libiq_venv310)
    - Installing required Python packages via pip

# 1. Run the Installer Script
Execute the following script from either outside or inside the libiq directory:
```
./libiq_installer.sh
```
This script handles all system-level and Python dependency setup automatically.

# 2. Activate the Python Virtual Environment
After installation, activate the environment:
```
source .libiq_venv310/bin/activate
```

# 3. Build the LibIQ Python Package
Once inside the virtual environment, build the project using:
```
hatch env create
hatch build
```
If hatch is not already installed, install it with:
```
pip install hatch
```
# 4. Install the Package
```
pip install dist/libiq-0.1.0.tar.gz
```