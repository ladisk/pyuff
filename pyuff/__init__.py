from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("pyuff")  # set in pyproject.toml
except PackageNotFoundError:  # imported from a source tree without installing
    __version__ = "unknown"

from .pyuff import *
from .datasets import *
