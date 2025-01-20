from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os

class CustomBuildExt(build_ext):
    def run(self):
        super().run()

module = Extension(
    'libiq._libiqwrapped',  # NB: "libiqwrapped" senza underscore iniziale
    sources=[
        "src/libiq_swig/libiq_wrapped.i",
        "src/libiq_swig/converter.cpp",
        "src/libiq_swig/analyzer.cpp"
    ],
    swig_opts=['-c++', '-python', '-outdir', 'src/libiq'],  # vedi nota sotto
    include_dirs=[
        "/usr/local/include",
        "/usr/local/include/sigmf",
        "./libs/libsigmf/external/flatbuffers/include",
        "./libs/libsigmf/external/json/include",
        "src/libiq_swig"
    ],
    # Aggiungiamo "fftw3_threads" per usare la versione multi-thread di FFTW
    libraries=["matio", "fftw3", "fftw3_threads"],
    library_dirs=["/usr/local/lib"],
    language='c++',
    # Aggiungiamo le opzioni per OpenMP e ottimizzazione
    extra_compile_args=['-fopenmp', '-O2', '-std=c++17'],
    extra_link_args=['-fopenmp']
)

setup(
    name='libiq',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[module],
    cmdclass={'build_ext': CustomBuildExt},
    zip_safe=False,
)
