# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Comparisons functions expected by the python array API

"""

from .utils import get_array_and_backend
from numbers import Number
import numpy as np

__all__ = (
    "all",    
    "allclose",
    "all_equal",
    "any",
    "isclose"
)


def _helper(x, fname, **kwargs):
    """
    Helper function to apply a comparison between
    """    
    if isinstance(x, Number):        
        if 'y' in kwargs:
            y = kwargs.pop('y')
            if isinstance(y, Number):
                fn = getattr(np, fname)
            else:
                y, backend_y = get_array_and_backend(y)
                fn = getattr(backend_y.array_namespace, fname)
            return fn(x, y, **kwargs)
        else: 
            return fn(x, **kwargs)
        
    x, backend_x = get_array_and_backend(x)
    fn = getattr(backend_x.array_namespace, fname)
    if 'y' in kwargs:
        y = kwargs.pop('y')
        if isinstance(y, Number):
            pass
        else:
            y, backend_y = get_array_and_backend(y)
            assert backend_x == backend_y, f"Two different backends {backend_x.impl} and {backend_y.impl} were provided, This operation is not supported by odl functions. Please ensure that your objects have the same implementation."
        return fn(x, y, **kwargs)
    else:
        return fn(x, **kwargs)

def all(x):
    """
    Test whether all array elements along a given axis evaluate to True.
    """
    return _helper(x, 'all')

def allclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns True if two arrays are element-wise equal within a tolerance.
    Note: This is not a Python Array API method, but it happens to work in Numpy and Pytorch.
    """
    return _helper(x, 'allclose', y=y, rtol=rtol, atol=atol, equal_nan=equal_nan)

def all_equal(x, y):
    """
    Test whether all array elements along a given axis evaluate to True.
    """
    return _helper(_helper(x, 'equal', y=y), 'all')

def any(x):
    """
    Test whether any array element along a given axis evaluates to True.
    """
    return _helper(x, 'any')

def isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.
    Note: This is not a Python Array API method, but it happens to work in Numpy and Pytorch.
    """
    return _helper(x, 'isclose', y=y, rtol=rtol, atol=atol, equal_nan=equal_nan)

