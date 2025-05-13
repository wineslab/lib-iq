# LibIQ Library
## Overview
LibIQ is a modular and extensible library designed for the manipulation, visualization, and classification of I/Q (In-phase and Quadrature) samples. It provides a Python interface built on a C++ backend using SWIG, enabling high performance for signal analysis tasks while maintaining ease of use in Python environments.
## Installation
### Prerequisites
Install the basic tools required to build the libraries:
```
sudo apt install git cmake g++ libtool graphviz swig -y

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev python3.10-tk python3-pip -y
```
### Initialize Git Submodules
```
git submodule update --init --recursive libs/libsigmf libs/zlib libs/hdf5
git submodule update --init libs/matio
```
### Build and Install Dependencies
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
### Make sure to use python 3.10
```
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2
```
Select the correct version of python
```
sudo update-alternatives --config python
```
Do the same procedure also for python3
```
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
```
Select the correct version of python3
```
sudo update-alternatives --config python3
```
If you want, there is also the possibility to create a python virtual environment by doing:
```
python3 -m venv venv310
```
and to access the virtual environment:
```
source venv310/bin/activate
```
finally when you are done with LibIQ, you can exit from the virtual environment by typing on the terminal
```
deactivate
```
### Build the LibIQ Python Package
If hatch is not already installed, install it with:
```
pip install build hatch
```
If hatch is installed, execute
```
hatch env create
hatch build
```
### Install the Package
```
pip install dist/libiq-0.1.0.tar.gz
```
## Automated Script for Build and Installation
We provide also a bash script that performs the steps descripted above automatically.

To execute the script, from inside the libiq directory do:
```
./libiq_installer.sh
```
This script handles all system-level and Python dependency setup automatically.
## After installation steps
It may be necessary to run `export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH` and `export PATH="$HOME/.local/bin:$PATH"`
for a temporary solution.

For a permanent solution instead do:
```
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
## Optional Features
### Profiling with ydata-profiling
Some utilities in LibIQ support generating detailed data profiling reports using the ydata-profiling library.

`Note: ydata-profiling depends on htmlmin==0.1.12, which uses an old setup.py format and may fail to build in system-wide installs.`

To avoid issues, we strongly recommend using a virtual environment (venv) for installing this optional dependency.

You can install LibIQ with profiling support like this:

If you do not already have built and installed libiq, do:
```
# Inside your virtual environment
pip install dist/libiq-0.1.0.tar.gz[profile]
```
If you already installed LibIQ and want to add profiling support later:
```
pip install 'libiq[profile]'
```
