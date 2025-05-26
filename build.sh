#!/bin/bash

set -e

path="$PWD"

SUDO=""
if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
fi

if command -v apt >/dev/null 2>&1; then
    echo "Detected apt-based system (e.g., Ubuntu)"
    $SUDO apt update
    $SUDO apt install -y graphviz swig wget make gcc g++ libtool automake
elif command -v yum >/dev/null 2>&1; then
    echo "Detected yum-based system (e.g., manylinux)"
    $SUDO yum update -y
    $SUDO yum install -y graphviz swig wget make gcc gcc-c++ libtool automake
else
    echo "Unsupported package manager: only apt or yum are supported"
    exit 1
fi

mkdir -p "$path/libs"
cd "$path/libs"
wget -O "fftw-3.3.10.tar.gz" https://fftw.org/fftw-3.3.10.tar.gz
tar -xzf "fftw-3.3.10.tar.gz" -C "./"
rm fftw-3.3.10.tar.gz

cd "$path/libs/fftw-3.3.10"
./configure --enable-shared --with-pic --enable-threads
make -j"$(nproc)"
$SUDO make install
$SUDO ldconfig

cd "$path"
rm -rf "$path/libs"

hatch build

echo "Success!"
