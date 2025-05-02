# LibIQ Library

## Overview
LibIQ is a modular and extensible library designed for the manipulation, visualization, and classification of I/Q (In-phase and Quadrature) samples. It provides a Python interface built on a C++ backend using SWIG, enabling high performance for signal analysis tasks while maintaining ease of use in Python environments.

## Automated Build and Installation
To build `libiq` the steps you need to follow are:

### 1. Run the Installer Script
Execute the following script from inside the libiq directory:
```
./libiq_installer.sh
```
This script handles all system-level and Python dependency setup automatically.

### 2. Activate the Python Virtual Environment
After installation, activate the environment:
```
source ./.libiq_venv310/bin/activate
```

### 3. Build the LibIQ Python Package
Once inside the virtual environment, build the project using:
If hatch is not already installed, install it with:
```
pip install build hatch
```
If hatch is installed, execute
```
hatch env create
hatch build
```

### 4. Install the Package
```
pip install dist/libiq-0.1.0.tar.gz
```

## Manual Installation
If you prefer manual control:

### Prerequisites
Install the basic tools required to build the libraries:
```
sudo apt install git cmake g++ libtool graphviz swig -y

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev python3.10-tk -y

```
### 1. Initialize Git Submodules
```
git submodule update --init --recursive libs/libsigmf libs/RFDataFactory libs/sdr_channelizer libs/zlib libs/hdf5 || echo "Problem initializing submodules"
git submodule update --init libs/matio
```
### 2. Build and Install Dependencies
Manually compile and install all C/C++ libraries in `libs/`. You must install them in standard locations such as `/usr/local/include`, so they can be found during compilation and linking.
Below are the installation steps for each dependency:
#### Install zlib
```
cd libs/zlib/
cmake .
cmake --build . --parallel $(nproc)
sudo cmake --install .
cd ../../
```
#### Install HDF5
```
cd libs/hdf5
./configure --prefix=/usr/local/include/hdf5 --enable-cxx
make -j$(nproc)
sudo make install
cd ../../
```
#### Install matio
```
cd libs/matio
./autogen.sh
./configure --enable-mat73=yes --with-default-file-ver=7.3 --with-hdf5="/usr/local/include/hdf5"
make -j$(nproc)
sudo make install PREFIX=/usr/local/include/matio
cd ../../
```
#### Install sigmf
```
cd libs/libsigmf
mkdir build && cd build
cmake ../
make -j$(nproc)
sudo make install
cd ../../../
```
#### Install FFTW
download from the official page of FFTW the fftw-3.3.10
```
wget -O "libs/fftw-3.3.10.tar.gz" https://fftw.org/fftw-3.3.10.tar.gz
tar -xzf "libs/fftw-3.3.10.tar.gz" -C "libs/"
rm libs/fftw-3.3.10.tar.gz
```
Build and install
```
cd libs/fftw-3.3.10
./configure --enable-shared --with-pic --enable-threads
make -j$(nproc)
sudo make install
cd ../../
```
#### After all libraries are installed, update the system’s dynamic linker cache:
```
sudo ldconfig
```

### 3. Create and activate a Python Virtual Environment
```
python3.10 -m venv .libiq_venv310
source ./.libiq_venv310/bin/activate
pip install --upgrade pip
```
Then continue from step 3 of the automatic installation section (Build the LibIQ Python Package).

It may be necessary to run `export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH` for a temporary solution or
```
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
for a permanent solution

