# Libiq Library


Libiq is a modular and extensible library designed for the manipulation, visualization, and classification of I/Q (In-phase and Quadrature) samples.

It is structured into four main classes:

- Analyzer  
  Provides tools for manipulating and analyzing time-series I/Q samples.  
  - Supports reading binary/CSV files and extracting real, imaginary, or complex components.  
  - Includes FFT and PSD methods for frequency-domain analysis.

- Plotter  
   Enables real-time visualization of I/Q signals through various plot types:  
  - Scatterplots:
    - Plot I vs Q to visualize signal characteristics.
    - Plot magnitude or phase over time to capture signal evolution.
  - Spectrograms: Compute and visualize time-varying frequency content using FFT. It supports custom window size and overlap settings for tuning resolution:
      - Smaller windows capture fine-grained variations.
      - Larger windows provide a broader overview of long-term signal behavior.

- Preprocessor  
  Handles I/Q data preprocessing for CNN training pipelines.  
  - Converts binary/CSV files into structured datasets.  
  - Implements energy peak detection to isolate RFI signals and ensure model generalization.

- Classifier  
  Contains methods to train and test a lightweight CNN model for RF signal classification. It uses real/imaginary parts, magnitude, and phase as input features.

Libiq has been successfully tested with python 3.9, 3.10,  3.11 and 3.12.

## Installation

There are two different ways to install Libiq

### Package repository installation

The easiest way to install libiq is through PyPI. Simply run:

```bash
pip install libiq
```

It also offers optional dependencies, such as ydata-profiling and scienceplot, which can be installed as follows:

To enable reporting features:

```bash
pip install libiq[report]
```

To enable enhanced plotting styles:

```bash
pip install libiq[styles]
```

Or, to install all optional features at once:

```bash
pip install libiq[all]
```

To verify that the library has been installed correctly, a test script is provided in the `docs` directory under the name `test_libiq.py`.

You can run it using pytest:

```bash
pytest -v test_libiq.py
```

### Build bash script

We provide a bash script that autmatically performs the steps described in [Source installation](#source-installation).

To run it you simply need to execute

```bash
sudo ./build.sh
```

### Source installation

#### Prerequisites

Install the basic tools required to build the libraries:

```bash
sudo apt install graphviz swig -y
```

#### Build and Install Dependencies

To work properly, Libiq needs [FFTW](https://www.fftw.org/index.html).

The installation steps for this library starts with the download of the sources from the official site of FFTW, in particular we need fftw-3.3.10

```bash
mkdir libs
wget -O "libs/fftw-3.3.10.tar.gz" https://fftw.org/fftw-3.3.10.tar.gz
tar -xzf "libs/fftw-3.3.10.tar.gz" -C "libs/"
rm libs/fftw-3.3.10.tar.gz
```

Then we build and install

```bash
cd libs/fftw-3.3.10
./configure --enable-shared --with-pic --enable-threads
make -j$(nproc)
sudo make install
cd ../../
sudo ldconfig
```

### Build the Libiq Python Package

This repository uses `hatch` for building the package.

```bash
hatch build
```

### Install the Package

```bash
pip install dist/libiq-*.tar.gz
```

Then if you want to install the optional dependencies, do as in [Package repository installation](#package-repository-installation)

If you use the libiq library to develop your own works, please cite the following paper:

```
@inproceedings{olimpieri2025libiq,
  author    = {Olimpieri, Filippo and Giustini, Noemi and Lacava, Andrea and D’Oro, Salvatore and Melodia, Tommaso and Cuomo, Francesca},
  title     = {{LibIQ: Toward Real-Time Spectrum Classification in O-RAN dApps}},
  booktitle = {Proceedings of the IEEE Mediterranean Communication and Computer Networking Conference (MedComNet)},
  year      = {2025},
  address   = {Cagliari, Italy},
  organization = {IEEE}
}
```

arxiv url: https://arxiv.org/abs/2505.10537