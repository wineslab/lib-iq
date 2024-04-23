#!/bin/bash

# Naviga alla directory src
cd src

# Per ogni file .i nella directory corrente
#for i in *.i; do
#    # Esegui il comando swig
#    swig -c++ -python "$i"
#done
swig -c++ -python "libiq.i"

# Ritorna alla directory precedente
cd ..
python3 setup.py build_ext --inplace
cd src
cp ./libiq.py ../examples/
cd ..

if [ -d "/root/prove_varie" ]; then
    cp /root/libiq-101/_libiq.cpython-310-x86_64-linux-gnu.so /root/prove_varie
    cp /root/libiq-101/src/libiq.py /root/prove_varie
fi

if [ -d "/root/demo" ]; then
    cp /root/libiq-101/_libiq.cpython-310-x86_64-linux-gnu.so /root/demo
    cp /root/libiq-101/src/libiq.py /root/demo
fi