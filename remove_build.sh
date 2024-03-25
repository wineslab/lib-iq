#!/bin/bash

# Rimuovi la cartella iq_samples_mat, se esiste
if [ -d "./examples/iq_samples_mat" ]; then
    echo "Rimozione della cartella iq_samples_mat..."
    rm -rf "./examples/iq_samples_mat"
else
    echo "La cartella iq_samples_mat non esiste."
fi

# Rimuovi la cartella iq_samples_sigmf, se esiste
if [ -d "./examples/iq_samples_sigmf" ]; then
    echo "Rimozione della cartella iq_samples_sigmf..."
    rm -rf "./examples/iq_samples_sigmf"
else
    echo "La cartella iq_samples_sigmf non esiste."
fi

# Rimuovi la cartella build, se esiste
if [ -d "build" ]; then
    echo "Rimozione della cartella build..."
    rm -rf build
else
    echo "La cartella build non esiste."
fi

# Rimuovi il modulo Python compilato, se esiste
if [ -f "_libiq.cpython-39-x86_64-linux-gnu.so" ]; then
    echo "Rimozione del modulo Python compilato..."
    rm "_libiq.cpython-39-x86_64-linux-gnu.so"
else
    echo "Il modulo Python compilato non esiste."
fi

#!/bin/bash

# Naviga alla directory src
cd src

# Per ogni file .cxx nella directory corrente
for cxx_file in *.cxx; do
    # Se il file è stato generato da SWIG
    if [[ $cxx_file == *_wrap.cxx ]]; then
        # Rimuovi il file .cxx
        echo "Rimozione del file $cxx_file generato da SWIG..."
        rm "$cxx_file"

        # Rimuovi il corrispondente file .py
        py_file="${cxx_file%_wrap.cxx}.py"
        if [ -f "$py_file" ]; then
            echo "Rimozione del file $py_file generato da SWIG..."
            rm "$py_file"
        else
            echo "Il file $py_file generato da SWIG non esiste."
        fi
    fi
done

# Ritorna alla directory precedente
cd ..
cd examples
echo "Rimozione del file libiq.py generato da SWIG che si trova dentro examples..."
rm libiq.py
cd ..

