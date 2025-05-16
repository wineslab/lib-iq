import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CustomBuildExt(build_ext):
    def run(self):
        super().run()

pwd = os.path.abspath(os.path.dirname(__file__))

module = Extension(
    'libiq._libiqwrapped',

    sources=[
        "src/libiq_swig/libiq_wrapped.i",
        "src/libiq_swig/converter.cpp",
        "src/libiq_swig/analyzer.cpp"
    ],

    swig_opts=[
        '-c++',
        '-python',
        '-outdir', 'src/libiq',
        f'-I{os.path.join(pwd, "src", "libiq_swig")}',
        f'-I{os.path.join(pwd, "libs", "libsigmf", "external", "flatbuffers", "include")}',
        f'-I{os.path.join(pwd, "libs", "libsigmf", "external", "json", "include")}'
    ],

    include_dirs=[
        "/usr/local/include",
        "/usr/local/include/sigmf",
        os.path.join(pwd, "libs", "libsigmf", "external", "flatbuffers", "include"),
        os.path.join(pwd, "libs", "libsigmf", "external", "json", "include"),
        os.path.join(pwd, "src", "libiq_swig"),
    ],

    libraries=["matio", "fftw3", "fftw3_threads"],
    library_dirs=["/usr/local/lib"],
    language='c++',
    extra_compile_args=['-fopenmp', '-O2', '-std=c++17'],
    extra_link_args=['-fopenmp']
)

setup(
    name='libiq',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[module],
    cmdclass={'build_ext': CustomBuildExt},
    zip_safe=False,
)