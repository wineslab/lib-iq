#!/bin/bash
clear
BASE_PATH="$(pwd)"
# Variabili principali
BASE_DIR="$BASE_PATH/libiq"
LIBS_DIR="$BASE_DIR/libs"

# Variabili per librerie
ZLIB_DIR="$LIBS_DIR/zlib"
HDF5_ARCHIVE="$LIBS_DIR/hdf5-1_14_3.tar.gz"
HDF5_DIR="$LIBS_DIR/hdf5-hdf5-1_14_3"
FFTW_ARCHIVE="$LIBS_DIR/fftw-3.3.10.tar.gz"
FFTW_DIR="$LIBS_DIR/fftw-3.3.10"
MATIO_DIR="$LIBS_DIR/matio"
LIBSIGMF_DIR="$LIBS_DIR/libsigmf"
IQ_CLUSTERING_DIR="$LIBS_DIR/iq_clustering"

# Percorsi di installazione
ZLIB_INCLUDE="/usr/local/include/zlib.h"
HDF5_INCLUDE="/usr/local/include/hdf5"
FFTW_INCLUDE="/usr/local/include/fftw3.h"
MATIO_INCLUDE="/usr/local/include/matio.h"
MATIO_PUBCONF_INCLUDE="/usr/local/include/matio_pubconf.h"
SIGMF_INCLUDE="/usr/local/include/sigmf"

# Percorsi binari
SWIG_BIN="swig"
CMAKE_BIN="cmake"
GPP_BIN="g++"
LIBTOOL_BIN="libtool"

VENV_DIR=".libiq_venv310"
PYTHON_VERSION="3.10"
PYTHON_BIN="python3.10"

# Numero di core per compilazione
NPROC=$(nproc)

# Funzione per controllare se un comando è disponibile
check_command() {
    if command -v "$1" &> /dev/null; then
        if [[ "$1" == "libtool" ]]; then
            version=$("$1" --version | head -n 1 | awk '{print $4}')
        else
            version=$(eval "$2")
        fi
        echo "$1 è già installato ($version). Salto l'installazione."
    else
        echo "Installing $1..."
        sudo apt install -y "$3" || { echo "Errore durante l'installazione di $1."; exit 1; }
    fi
}

# Funzione per controllare se un pacchetto Python è installato
check_python_package() {
    if "$PYTHON_BIN" -m pip show "$1" &> /dev/null; then
        echo "Il pacchetto '$1' è già installato."
    else
        echo "Installazione del pacchetto '$1'..."
        "$PYTHON_BIN" -m pip install "$1" || { echo "Errore durante l'installazione di '$1'."; exit 1; }
    fi
}

# Funzione per controllare e installare pacchetti di sistema
install_package() {
    if dpkg -l | grep -qw "$1"; then
        echo "$1 è già installato."
    else
        echo "Installazione di $1..."
        sudo apt install -y "$1" || { echo "Errore durante l'installazione di $1."; exit 1; }
    fi
}

if [ -d "$BASE_DIR" ]; then
    echo "La directory $BASE_DIR esiste già. Salto git clone e l'inizializzazione dei submodules."
else
    echo "Clonazione del repository libiq..."
    git clone --branch iq_clustering https://github.com/wineslab/libiq.git "$BASE_DIR" || { echo "Errore durante il clonaggio di libiq."; exit 1; }

    # Entrare nella directory principale
    echo "Entering $BASE_DIR/"
    cd "$BASE_DIR" || { echo "Directory $BASE_DIR non trovata!"; exit 1; }

    # Aggiornare i submodules
    echo "Updating submodules di libiq in $BASE_DIR"
    git submodule update --init --recursive libs/libsigmf libs/RFDataFactory libs/sdr_channelizer libs/zlib || echo "Problema nell'inizializzazione dei submodules"
    git submodule update --init libs/matio
fi

# Controllo e installazione di Python 3.10
if command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Python $PYTHON_VERSION è già installato ($(python3.10 --version))."
else
    echo "Python $PYTHON_VERSION non trovato. Installazione in corso..."
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    install_package "python3.10"
    install_package "python3.10-venv"
    install_package "python3.10-dev"
    install_package "python3.10-distutils"
    install_package "python3.10-tk"
fi

install_package "graphviz"

# Navigare nella directory principale
cd "$BASE_DIR" || { echo "Directory $BASE_DIR non trovata!"; exit 1; }

# Creazione dell'ambiente virtuale solo se non esiste
if [ -d "$VENV_DIR" ]; then
    echo "L'ambiente virtuale $VENV_DIR esiste già. Lo attivo."
else
    echo "Creazione dell'ambiente virtuale $VENV_DIR con Python $PYTHON_VERSION..."
    "$PYTHON_BIN" -m venv "$VENV_DIR" || { echo "Errore durante la creazione dell'ambiente virtuale."; exit 1; }
fi

# Attivare l'ambiente virtuale
echo "Attivazione dell'ambiente virtuale $VENV_DIR..."
source "$VENV_DIR/bin/activate" || { echo "Errore durante l'attivazione dell'ambiente virtuale."; exit 1; }

# Aggiornamento di pip e installazione dei requirements
echo "Aggiornamento di pip..."
pip install --upgrade pip || { echo "Errore durante l'aggiornamento di pip."; exit 1; }

#./apply_setup.sh

#echo "iq_clustering installed successfully!"

check_python_package "matplotlib"

# Controllo per SWIG
check_command "$SWIG_BIN" "swig -version | grep 'SWIG Version' | awk '{print $3}'" "swig"

# Controllo per CMake
check_command "$CMAKE_BIN" "cmake --version | head -n 1 | awk '{print $3}'" "cmake"

# Controllo per g++
check_command "$GPP_BIN" "g++ --version | head -n 1 | awk '{print $4}'" "g++"

# Controllo per libtool e libtool-bin
if command -v "$LIBTOOL_BIN" &> /dev/null; then
    version=$("$LIBTOOL_BIN" --version | head -n 1 | awk '{print $4}')
    echo "$LIBTOOL_BIN è già installato ($version). Salto l'installazione."
else
    echo "Installing libtool and libtool-bin..."
    sudo apt install -y libtool libtool-bin || { echo "Errore durante l'installazione di libtool."; exit 1; }
fi

# Controllo se zlib è già installato
if [ -f "$ZLIB_INCLUDE" ]; then
    echo "zlib è già presente in /usr/local/include/. Salto la compilazione e installazione."
else
    echo "Entering $ZLIB_DIR/"
    cd "$ZLIB_DIR" || { echo "Directory $ZLIB_DIR non trovata!"; exit 1; }

    echo "Building zlib..."
    cmake . || { echo "Errore durante la configurazione di CMake."; exit 1; }
    cmake --build . --parallel "$NPROC" || { echo "Errore durante la compilazione di zlib."; exit 1; }

    echo "Installing zlib in /usr/local/include/"
    sudo cmake --install . || { echo "Errore durante l'installazione di zlib."; exit 1; }

    echo "Installation of zlib completed successfully!"
fi

# Controllo se HDF5 è già installato
if [ -d "$HDF5_INCLUDE" ]; then
    echo "HDF5 è già installato in /usr/local/include/. Salto la configurazione, compilazione e installazione."
else
    if [ -d "$HDF5_DIR" ]; then
        echo "La directory $HDF5_DIR è già presente. Salto il download e lo spacchettamento."
    else
        if [ -f "$HDF5_ARCHIVE" ]; then
            echo "Il file $HDF5_ARCHIVE è già presente. Salto il download."
        else
            echo "Downloading HDF5..."
            wget -O "$HDF5_ARCHIVE" https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz || { echo "Errore durante il download di HDF5."; exit 1; }
        fi

        echo "Unpacking $HDF5_ARCHIVE"
        tar -xzf "$HDF5_ARCHIVE" -C "$LIBS_DIR" || { echo "Errore durante lo spacchettamento di HDF5."; exit 1; }
        rm $HDF5_ARCHIVE
    fi

    echo "Entering $HDF5_DIR/"
    cd "$HDF5_DIR" || { echo "Directory $HDF5_DIR non trovata!"; exit 1; }

    echo "Starting HDF5 configuration..."
    ./configure --prefix=/usr/local/include/hdf5 --enable-cxx
    make -j"$NPROC"
    sudo make install
    echo "HDF5 installed successfully"
fi

# Controllo se matio è già installato
if [ -f "$MATIO_INCLUDE" ] || [ -f "$MATIO_PUBCONF_INCLUDE" ]; then
    echo "matio è già installato in /usr/local/include/. Salto la configurazione, compilazione e installazione."
else
    echo "Entering $MATIO_DIR/"
    cd "$MATIO_DIR" || { echo "Directory $MATIO_DIR non trovata!"; exit 1; }

    echo "Eseguendo autogen.sh..."
    ./autogen.sh || { echo "Errore durante l'esecuzione di autogen.sh."; exit 1; }
    
    echo "Configurazione di matio..."
    ./configure --enable-mat73=yes --with-default-file-ver=7.3 --with-hdf5="$HDF5_INCLUDE" || { echo "Errore durante la configurazione di matio."; exit 1; }

    echo "Compilazione di matio..."
    make -j"$NPROC" || { echo "Errore durante la compilazione di matio."; exit 1; }

    echo "Installazione di matio..."
    sudo make install PREFIX=/usr/local/include/matio || { echo "Errore durante l'installazione di matio."; exit 1; }

    echo "matio installed successfully!"
fi

# Controllo se libsigmf è già installato
if [ -d "$SIGMF_INCLUDE" ]; then
    echo "libsigmf è già installato in /usr/local/include/. Salto la compilazione e installazione."
else
    echo "Entering $LIBSIGMF_DIR/"
    cd "$LIBSIGMF_DIR" || { echo "Directory $LIBSIGMF_DIR non trovata!"; exit 1; }

    if [ ! -d "build" ]; then
        echo "Creating build directory..."
        mkdir build || { echo "Errore durante la creazione della directory build."; exit 1; }
    else
        echo "La directory build esiste già. Utilizzo la directory esistente."
    fi

    cd build || { echo "Directory build non trovata!"; exit 1; }

    echo "Configuring libsigmf with CMake..."
    cmake ../ || { echo "Errore durante la configurazione con CMake."; exit 1; }

    echo "Building libsigmf..."
    make -j"$NPROC" || { echo "Errore durante la compilazione di libsigmf."; exit 1; }

    echo "Installing libsigmf..."
    sudo make install || { echo "Errore durante l'installazione di libsigmf."; exit 1; }

    echo "libsigmf installed successfully!"
fi

# Controllo se FFTW è già installato
if [ -f "$FFTW_INCLUDE" ]; then
    echo "FFTW è già installato in /usr/local/include/. Salto la configurazione, compilazione e installazione."
else
    if [ -d "$FFTW_DIR" ]; then
        echo "La directory $FFTW_DIR è già presente. Salto il download e lo spacchettamento."
    else
        if [ -f "$FFTW_ARCHIVE" ]; then
            echo "Il file $FFTW_ARCHIVE è già presente. Salto il download."
        else
            echo "Downloading FFTW..."
            wget -O "$FFTW_ARCHIVE" https://fftw.org/fftw-3.3.10.tar.gz || { echo "Errore durante il download di FFTW."; exit 1; }
        fi

        echo "Unpacking $FFTW_ARCHIVE"
        tar -xzf "$FFTW_ARCHIVE" -C "$LIBS_DIR" || { echo "Errore durante lo spacchettamento di FFTW."; exit 1; }
        rm "$FFTW_ARCHIVE"
    fi

    echo "Entering $FFTW_DIR/"
    cd "$FFTW_DIR" || { echo "Directory $FFTW_DIR non trovata!"; exit 1; }

    echo "Starting FFTW configuration..."
    ./configure --enable-shared --with-pic --enable-threads || { echo "Errore durante la configurazione di FFTW."; exit 1; }

    echo "Building FFTW..."
    make -j"$NPROC" || { echo "Errore durante la compilazione di FFTW."; exit 1; }

    echo "Installing FFTW..."
    sudo make install || { echo "Errore durante l'installazione di FFTW."; exit 1; }

    echo "FFTW installed successfully!"
fi

# Controllo se GNURadio è già installato
if command -v gnuradio-config-info &> /dev/null; then
    version=$(gnuradio-config-info --version)
    echo "GNURadio è già installato (Versione: $version). Salto l'installazione."
else
    echo "Installing GNURadio..."
    sudo apt install gnuradio -y || { echo "Errore durante l'installazione di GNURadio."; exit 1; }
fi

sudo ldconfig