#!/bin/bash

mkdir -p libiq_build/build

cd src
swig -c++ -python "libiq.i"
cd ..

python3 setup.py build_ext --build-lib libiq_build --build-temp libiq_build/build

cd src
cp ./libiq.py ../examples/
cp ./libiq.py ../libiq_build/
cd ..

cd libiq_build
for file in _libiq.cpython*; do
    if [ -f "$file" ]; then
        cp "$file" ../examples/
    fi
done