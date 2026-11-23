"""Builds the optional compiled parser pyuff/_fastparse (Cython).

    pip install cython setuptools numpy
    python setup_fastparse.py

pyuff works without it (NumPy is used instead); with it, ASCII data-set 58 is read faster.
The build runs in a temporary directory, so that setuptools does not read pyproject.toml
(the package itself is built by hatchling); the module is written to pyuff/.
"""
import os
import sys
import tempfile

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

root = os.path.dirname(os.path.abspath(__file__))
source = os.path.join(root, 'pyuff', '_fastparse.pyx')

with tempfile.TemporaryDirectory() as tmp:
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        setup(
            name='pyuff-fastparse',
            ext_modules=cythonize(
                [Extension('pyuff._fastparse', [source], include_dirs=[np.get_include()],
                           define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])],
                language_level=3, build_dir=tmp),
            script_args=['build_ext', '--build-lib', root, '--build-temp', tmp] + sys.argv[1:],
        )
    finally:
        os.chdir(cwd)
