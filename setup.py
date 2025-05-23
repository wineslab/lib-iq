import os
import re

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py as _build_py


class CustomBuildExt(build_ext):
    def run(self):
        super().run()


class CustomBuildPy(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()


pwd = os.path.abspath(os.path.dirname(__file__))
local_lib = "/usr/local/lib"

with open("VERSION", "r") as f:
    version = f.read().strip()

version_pattern = r"^\d+\.\d+\.\d+$"
if not re.match(version_pattern, version):
    raise ValueError(
        f"[ERROR] Invalid version format: '{version}'. Expected format is MAJOR.MINOR.PATCH (e.g., 1.0.0)"
    )

module = Extension(
    "libiq._libiqwrapped",
    sources=[
        "src/libiq_swig/libiq_wrapped.i",
        "src/libiq_swig/analyzer.cpp",
    ],
    swig_opts=["-c++", "-outdir", "src/libiq", "-Isrc/libiq_swig"],
    include_dirs=[
        "/usr/local/include",
        os.path.join(pwd, "src", "libiq_swig"),
    ],
    extra_objects=[
        os.path.join(local_lib, "libfftw3.a"),
        os.path.join(local_lib, "libfftw3_threads.a"),
    ],
    library_dirs=["/usr/local/lib"],
    language="c++",
    extra_compile_args=["-fopenmp", "-O2", "-std=c++17"],
    extra_link_args=["-fopenmp"],
)

setup(
    name="libiq",
    version=version,
    python_requires=">=3.9,<3.13",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[module],
    cmdclass={
        "build_ext": CustomBuildExt,
        "build_py": CustomBuildPy,
    },
    package_data={
        "libiq": ["libiqwrapped.py"],
    },
    zip_safe=False,
)
