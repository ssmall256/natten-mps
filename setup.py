"""
Optional setup.py for building the nanobind Metal extension.

Usage:
    pip install -e ".[nanobind]"    # build with nanobind extension
    pip install -e .                # pure Python only (no compilation)
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir / 'natten_mps' / '_core'}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE=Release",
        ]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", "--config", "Release", "-j"],
            cwd=build_temp,
            check=True,
        )


# Only add CMake extension if nanobind is available
try:
    import nanobind
    ext_modules = [CMakeExtension("natten_mps._core._nanobind_ext")]
    cmdclass = {"build_ext": CMakeBuild}
except ImportError:
    ext_modules = []
    cmdclass = {}


setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
