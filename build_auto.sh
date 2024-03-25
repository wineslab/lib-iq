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
python setup.py build_ext --inplace
cd src
cp ./libiq.py ../examples/
cd ..