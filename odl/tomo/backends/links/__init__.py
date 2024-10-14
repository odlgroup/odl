from . import base
from . import numpy

from .base import (
    are_compatible
)


def _is_package_available(package):
    import importlib

    try:
        importlib.import_module(package)
        return True
    except ModuleNotFoundError:
        return False


# Import torch linking facility if torch is available.
if _is_package_available("torch"):
    from . import torch

# Import cupy linking facility if cupy is available.
if _is_package_available("cupy"):
    from . import cupy
