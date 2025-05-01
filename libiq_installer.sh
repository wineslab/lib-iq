#!/bin/bash
clear
BASE_PATH="$(pwd)"

if [[ "$(basename "$BASE_PATH")" == "libiq" ]]; then
    BASE_DIR="$BASE_PATH"
    LIBS_DIR="$BASE_DIR/libs"
    
    echo "Already inside libiq. Skipping clone."
    
    echo "Switching to 'libiq_clean' branch..."
    git switch libiq_clean || { echo "Failed to switch to libiq_clean branch."; exit 1; }
    git pull origin libiq_clean || { echo "Failed to pull updates from libiq_clean branch."; exit 1; }

    echo "Updating submodules..."
    git submodule update --init --recursive libs/libsigmf libs/RFDataFactory libs/sdr_channelizer libs/zlib || echo "Problem initializing submodules"
    git submodule update --init libs/matio
else
    BASE_DIR="$BASE_PATH/libiq"
    LIBS_DIR="$BASE_DIR/libs"

    if [ -d "$BASE_DIR" ]; then
        echo "The directory $BASE_DIR already exists. Skipping git clone and submodules."
    else
        echo "Cloning libiq repository..."
        git clone --branch libiq_clean https://github.com/wineslab/libiq.git "$BASE_DIR" || { echo "Error cloning libiq."; exit 1; }

        echo "Entering $BASE_DIR/"
        cd "$BASE_DIR" || { echo "Directory $BASE_DIR not found!"; exit 1; }

        echo "Updating submodules in $BASE_DIR"
        git submodule update --init --recursive libs/libsigmf libs/RFDataFactory libs/sdr_channelizer libs/zlib || echo "Problem initializing submodules"
        git submodule update --init libs/matio
    fi
fi

ZLIB_DIR="$LIBS_DIR/zlib"
HDF5_ARCHIVE="$LIBS_DIR/hdf5-1_14_3.tar.gz"
HDF5_DIR="$LIBS_DIR/hdf5-hdf5-1_14_3"
FFTW_ARCHIVE="$LIBS_DIR/fftw-3.3.10.tar.gz"
FFTW_DIR="$LIBS_DIR/fftw-3.3.10"
MATIO_DIR="$LIBS_DIR/matio"
LIBSIGMF_DIR="$LIBS_DIR/libsigmf"

ZLIB_INCLUDE="/usr/local/include/zlib.h"
HDF5_INCLUDE="/usr/local/include/hdf5"
FFTW_INCLUDE="/usr/local/include/fftw3.h"
MATIO_INCLUDE="/usr/local/include/matio.h"
MATIO_PUBCONF_INCLUDE="/usr/local/include/matio_pubconf.h"
SIGMF_INCLUDE="/usr/local/include/sigmf"

SWIG_BIN="swig"
CMAKE_BIN="cmake"
GPP_BIN="g++"
LIBTOOL_BIN="libtool"

VENV_DIR=".libiq_venv310"
PYTHON_VERSION="3.10"
PYTHON_BIN="python3.10"

NPROC=$(nproc)

check_command() {
    if command -v "$1" &> /dev/null; then
        if [[ "$1" == "libtool" ]]; then
            version=$("$1" --version | head -n 1 | awk '{print $4}')
        else
            version=$(eval "$2")
        fi
        echo "$1 is already installed ($version). Skipping installation."
    else
        echo "Installing $1..."
        sudo apt install -y "$3" || { echo "Error installing $1."; exit 1; }
    fi
}

check_python_package() {
    if "$PYTHON_BIN" -m pip show "$1" &> /dev/null; then
        echo "Package '$1' is already installed."
    else
        echo "Installing package '$1'..."
        "$PYTHON_BIN" -m pip install "$1" || { echo "Error installing package '$1'."; exit 1; }
    fi
}

install_package() {
    if dpkg -l | grep -qw "$1"; then
        echo "$1 is already installed."
    else
        echo "Installing $1..."
        sudo apt install -y "$1" || { echo "Error installing $1."; exit 1; }
    fi
}

if command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Python $PYTHON_VERSION is already installed ($($PYTHON_BIN --version))."
else
    echo "Python $PYTHON_VERSION not found. Installing..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    install_package "python3.10"
    install_package "python3.10-venv"
    install_package "python3.10-dev"
    install_package "python3.10-distutils"
    install_package "python3.10-tk"
fi

install_package "graphviz"

cd "$BASE_DIR" || { echo "Directory $BASE_DIR not found!"; exit 1; }

if [ -d "$VENV_DIR" ]; then
    echo "The virtual environment $VENV_DIR already exists. Activating it."
else
    echo "Creating virtual environment $VENV_DIR with Python $PYTHON_VERSION..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" || { echo "Error while creating the virtual environment."; exit 1; }
fi

echo "Activating virtual environment $VENV_DIR..."
source "$VENV_DIR/bin/activate" || { echo "Error while activating the virtual environment."; exit 1; }

echo "Upgrading pip..."
pip install --upgrade pip || { echo "Error while upgrading pip."; exit 1; }

check_python_package "matplotlib"
check_python_package "hatch"

check_command "$SWIG_BIN" "swig -version | grep 'SWIG Version' | awk '{print $3}'" "swig"

check_command "$CMAKE_BIN" "cmake --version | head -n 1 | awk '{print $3}'" "cmake"

check_command "$GPP_BIN" "g++ --version | head -n 1 | awk '{print $4}'" "g++"

if command -v "$LIBTOOL_BIN" &> /dev/null; then
    version=$("$LIBTOOL_BIN" --version | head -n 1 | awk '{print $4}')
    echo "$LIBTOOL_BIN is already installed (version $version). Skipping installation."
else
    echo "Installing libtool and libtool-bin..."
    sudo apt install -y libtool libtool-bin || { echo "Error installing libtool."; exit 1; }
fi

if [ -f "$ZLIB_INCLUDE" ]; then
    echo "zlib is already present in /usr/local/include/. Skipping build and installation."
else
    echo "Entering $ZLIB_DIR/"
    cd "$ZLIB_DIR" || { echo "Directory $ZLIB_DIR not found!"; exit 1; }

    echo "Building zlib..."
    cmake . || { echo "Error during CMake configuration."; exit 1; }
    cmake --build . --parallel "$NPROC" || { echo "Error during zlib compilation."; exit 1; }

    echo "Installing zlib to /usr/local/include/"
    sudo cmake --install . || { echo "Error during zlib installation."; exit 1; }

    echo "zlib installation completed successfully!"
fi

if [ -d "$HDF5_INCLUDE" ]; then
    echo "HDF5 is already installed in /usr/local/include/. Skipping configuration, build, and installation."
else
    if [ -d "$HDF5_DIR" ]; then
        echo "The directory $HDF5_DIR already exists. Skipping download and extraction."
    else
        if [ -f "$HDF5_ARCHIVE" ]; then
            echo "The file $HDF5_ARCHIVE already exists. Skipping download."
        else
            echo "Downloading HDF5..."
            wget -O "$HDF5_ARCHIVE" https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz \
                || { echo "Error downloading HDF5."; exit 1; }
        fi

        echo "Unpacking $HDF5_ARCHIVE"
        tar -xzf "$HDF5_ARCHIVE" -C "$LIBS_DIR" \
            || { echo "Error unpacking HDF5."; exit 1; }
        rm "$HDF5_ARCHIVE"
    fi

    echo "Entering $HDF5_DIR/"
    cd "$HDF5_DIR" || { echo "Directory $HDF5_DIR not found!"; exit 1; }

    echo "Starting HDF5 configuration..."
    ./configure --prefix=/usr/local/include/hdf5 --enable-cxx
    make -j"$NPROC"
    sudo make install
    echo "HDF5 installed successfully"
fi

if [ -f "$MATIO_INCLUDE" ] || [ -f "$MATIO_PUBCONF_INCLUDE" ]; then
    echo "matio is already installed in /usr/local/include/. Skipping configuration, build, and installation."
else
    echo "Entering $MATIO_DIR/"
    cd "$MATIO_DIR" || { echo "Directory $MATIO_DIR not found!"; exit 1; }

    echo "Running autogen.sh..."
    ./autogen.sh || { echo "Error running autogen.sh."; exit 1; }
    
    echo "Configuring matio..."
    ./configure --enable-mat73=yes --with-default-file-ver=7.3 --with-hdf5="$HDF5_INCLUDE" \
        || { echo "Error configuring matio."; exit 1; }

    echo "Building matio..."
    make -j"$NPROC" || { echo "Error building matio."; exit 1; }

    echo "Installing matio..."
    sudo make install PREFIX=/usr/local/include/matio \
        || { echo "Error installing matio."; exit 1; }

    echo "matio installed successfully!"
fi

if [ -d "$SIGMF_INCLUDE" ]; then
    echo "libsigmf is already installed in /usr/local/include/. Skipping build and installation."
else
    echo "Entering $LIBSIGMF_DIR/"
    cd "$LIBSIGMF_DIR" || { echo "Directory $LIBSIGMF_DIR not found!"; exit 1; }

    if [ ! -d "build" ]; then
        echo "Creating build directory..."
        mkdir build || { echo "Error creating build directory."; exit 1; }
    else
        echo "Build directory already exists. Using existing directory."
    fi

    cd build || { echo "Build directory not found!"; exit 1; }

    echo "Configuring libsigmf with CMake..."
    cmake ../ || { echo "Error configuring with CMake."; exit 1; }

    echo "Building libsigmf..."
    make -j"$NPROC" || { echo "Error building libsigmf."; exit 1; }

    echo "Installing libsigmf..."
    sudo make install || { echo "Error installing libsigmf."; exit 1; }

    echo "libsigmf installed successfully!"
fi

if [ -f "$FFTW_INCLUDE" ]; then
    echo "FFTW is already installed in /usr/local/include/. Skipping configuration, build, and installation."
else
    if [ -d "$FFTW_DIR" ]; then
        echo "Directory $FFTW_DIR already exists. Skipping download and extraction."
    else
        if [ -f "$FFTW_ARCHIVE" ]; then
            echo "File $FFTW_ARCHIVE already exists. Skipping download."
        else
            echo "Downloading FFTW..."
            wget -O "$FFTW_ARCHIVE" https://fftw.org/fftw-3.3.10.tar.gz || { echo "Error downloading FFTW."; exit 1; }
        fi

        echo "Extracting $FFTW_ARCHIVE"
        tar -xzf "$FFTW_ARCHIVE" -C "$LIBS_DIR" || { echo "Error extracting FFTW."; exit 1; }
        rm "$FFTW_ARCHIVE"
    fi

    echo "Entering $FFTW_DIR/"
    cd "$FFTW_DIR" || { echo "Directory $FFTW_DIR not found!"; exit 1; }

    echo "Configuring FFTW..."
    ./configure --enable-shared --with-pic --enable-threads || { echo "Error configuring FFTW."; exit 1; }

    echo "Building FFTW..."
    make -j"$NPROC" || { echo "Error building FFTW."; exit 1; }

    echo "Installing FFTW..."
    sudo make install || { echo "Error installing FFTW."; exit 1; }

    echo "FFTW installed successfully!"
fi

sudo ldconfig