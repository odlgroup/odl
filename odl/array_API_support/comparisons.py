from .utils import get_array_and_backend
from numbers import Number
import numpy as np

__all__ = (
    "all",    
    "allclose",
    "all_equal",
    "any",
    "asarray",
    "isclose"
)


def _helper(x, fname, **kwargs):    
    if isinstance(x, Number):        
        if 'y' in kwargs:
            y = kwargs.pop('y')
            if isinstance(y, Number):
                fn = getattr(np, fname)
            else:
                y, backend_y = get_array_and_backend(y)
                fn = getattr(backend_y, fname)
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

def asarray(x):
    """
    Returns an array corresponding to an ODL object.
    Note:
        This does not actually performs a comparison, yet it is located in this module for technical reasons due to the underlying helper function.
    """
    return _helper(x, 'asarray')

def isclose(x, y, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.
    Note: This is not a Python Array API method, but it happens to work in Numpy and Pytorch.
    """
    return _helper(x, 'isclose', y=y, rtol=rtol, atol=atol, equal_nan=equal_nan)

