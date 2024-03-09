import os
import re
import subprocess
import sys
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# This dictionary maps Python's platform names to CMake's platform names.
# It is used to set the correct architecture when building the extension with CMake.
PLAT_TO_CMAKE = {
    "win32": "Win32",       # For 32-bit Windows platforms
    "win-amd64": "x64",     # For 64-bit Windows platforms
    "win-arm32": "ARM",     # For 32-bit ARM Windows platforms
    "win-arm64": "ARM64",   # For 64-bit ARM Windows platforms
}

# Define a class for the CMake extension that inherits from setuptools.Extension
class CMakeExtension(Extension):
    # The constructor takes two parameters: the name of the extension and the source directory
    def __init__(self, name: str, sourcedir: str = "") -> None:
        # Call the constructor of the parent class
        super().__init__(name, sources=[])
        # Convert the source directory path to a string and resolve it to an absolute path
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

# Define a class for building the CMake extension that inherits from setuptools.command.build_ext
class CMakeBuild(build_ext):
    # Define a method for building the extension
    def build_extension(self, ext: CMakeExtension) -> None:
        # Get the full path of the extension
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        # Get the parent directory of the extension
        extdir = ext_fullpath.parent.resolve()

        # Determine the build configuration (Debug/Release) based on the DEBUG environment variable
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Get the CMake generator from the environment variables
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Define the CMake arguments
        cmake_args = [
            # Set the output directory for the library
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            # Set the Python executable to be used
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            # Set the build type (Debug/Release)
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        # Define the build arguments
        build_args = []

        # If there are additional CMake arguments in the environment variables, add them
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Add the version info to the CMake arguments
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        # If the compiler is not MSVC
        if self.compiler.compiler_type != "msvc":
            # If the CMake generator is not specified or is Ninja
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    # Try to import the ninja module
                    import ninja

                    # Get the path of the ninja executable
                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    # Add the Ninja generator and the path of the ninja executable to the CMake arguments
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass
        else:
            # If the compiler is MSVC
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # Check if the CMake generator contains an architecture
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # If the CMake generator does not specify a single configuration and does not contain an architecture
            if not single_config and not contains_arch:
                # Add the architecture to the CMake arguments
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # If the CMake generator does not specify a single configuration
            if not single_config:
                # Add the output directory to the CMake arguments
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                # Add the configuration to the build arguments
                build_args += ["--config", cfg]

        # If the platform is macOS
        if sys.platform.startswith("darwin"):
            # Get the architectures from the environment variables
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                # Add the architectures to the CMake arguments
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # If the parallel level is not specified in the environment variables
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # If the parallel attribute is set
            if hasattr(self, "parallel") and self.parallel:
                # Add the parallel level to the build arguments
                build_args += [f"-j{self.parallel}"]

        # Get the temporary build directory
        build_temp = Path(self.build_temp) / ext.name
        # If the temporary build directory does not exist
        if not build_temp.exists():
            # Create the temporary build directory
            build_temp.mkdir(parents=True)

        # Run CMake
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        # Build the project
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )

# Call the setup function to build and install the Python package
from setuptools import setup, find_packages

setup(
    # The name of the package
    name="libiq",
    # The version of the package
    version="0.0.1",
    # The name of the author(s)
    author="Filippo / Noemi",
    # The email of the author(s)
    author_email="olimpieri.1933529@studenti.uniroma1.it / giustini.1933541@studenti.uniroma1.it",
    # A short description of the package
    description="ORan security software",
    # A long description of the package. This is typically read from README.md.
    long_description="",
    # A list of all Python extensions to be built
    ext_modules=[CMakeExtension("libiq")],
    # A dictionary mapping command names to Command subclasses
    cmdclass={"build_ext": CMakeBuild},
    # If True, the package can be built as a .zip file. If False (the case here), the package will be built as a .tar.gz file.
    zip_safe=False,
    # A dictionary mapping extras (optional features of the project) to the Python packages that provide them.
    extras_require={"test": ["pytest>=8.0.1"]},
    # The Python version(s) required to install the package.
    python_requires=">=3.9.0",
    # The packages that will be installed
    install_requires=[
        'scipy',
        'tqdm',
    ],
    # The Python packages included in the package
    py_modules=['RFDataFactory.SigMF.sigmf_converter'],
)

