#!/bin/bash

set -e

SUDO=''
if [[ $EUID -ne 0 ]]; then
  SUDO='sudo'
fi

export PATH="$HOME/.local/bin:$PATH"

path="$PWD"

$SUDO apt update
$SUDO apt install -y graphviz swig

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
