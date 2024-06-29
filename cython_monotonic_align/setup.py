# modified from https://github.com/jaywalnut310/glow-tts/blob/master/monotonic_align/setup.py

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'cython_monotonic_align',
  ext_modules = cythonize("core.pyx"),
  include_dirs=[numpy.get_include()]
)