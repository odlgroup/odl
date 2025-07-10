from .utils import get_array_and_backend, lookup_array_backend
from numbers import Number
import numpy as np

__all__ = (
    'arange',
    'asarray',
    'empty',
    # 'eye',
    'from_dlpack',
    # 'full',
    'full_like',
    # 'linspace',
    # 'meshgrid',
    'ones',
    'ones_like',
    'tril',
    'triu',
    'zeros',
    'zeros_like'
)

def _helper_from_array(fname, x, **kwargs):    
    x, backend_x = get_array_and_backend(x)
    fn = getattr(backend_x.array_namespace, fname)
    return fn(x, **kwargs)

def _helper_from_shape(fname, impl, shape, **kwargs):    
    backend = lookup_array_backend(impl)
    fn = getattr(backend.array_namespace, fname)
    return fn(shape, **kwargs)

def arange(impl, start, stop=None, step=1, dtype=None, device=None):
    """
    Returns evenly spaced values within the half-open interval [start, stop) as a one-dimensional array.
    """
    backend = lookup_array_backend(impl)
    fn = getattr(backend.array_namespace, 'arange')
    return fn(start, stop=stop, step=step, dtype=dtype, device=device)

def asarray(x):
    """
    Returns an array corresponding to an ODL object.
    Note:
        This does not actually performs a comparison, yet it is located in this module for technical reasons due to the underlying helper function.
    """
    return _helper_from_array('asarray', x)

def empty(impl, shape, dtype=None, device=None):
    """
    Returns an uninitialized array having a specified shape.
    """
    return _helper_from_shape('empty', impl, shape=shape, dtype=dtype, device=device)

def empty_like(x, dtype=None, device=None):
    """
    Returns an uninitialized array with the same shape as an input array x.
    """
    return _helper_from_array('empty_like', x=x, dtype=dtype, device=device)

# def eye(n_rows, n_cols=None, k=0, dtype=None, device=None):
#     """
#     Returns a two-dimensional array with ones on the kth diagonal and zeros elsewhere.
#     """
#     return _helper('eye', n_rows=n_rows, n_cols=n_cols, k=k, dtype=dtype, device=device)

def from_dlpack(x, dtype=None, device=None):
    """
    Returns a new array containing the data from another (array) object with a __dlpack__ method.
    """
    return _helper_from_array('from_dlpack', x=x, dtype=dtype, device=device)

# def full(shape, fill_value, dtype=None, device=None):
#     """
#     Returns a new array having a specified shape and filled with fill_value.
#     """
#     return _helper('full', shape=shape, fill_value=fill_value, dtype=dtype, device=device)

def full_like(x, dtype=None, device=None):
    """
    Returns a new array filled with fill_value and having the same shape as an input array x.
    """
    return _helper_from_array('full_like', x=x, dtype=dtype, device=device)

# def linspace(start, stop, num, dtype=None, device=None, endpoint=True):
#     """
#     Returns evenly spaced numbers over a specified interval.
#     """
#     return _helper('linspace', start=start, stop=stop, num=num, dtype=dtype, device=device, endpoint=endpoint)

# def meshgrid(*arrays, indexing='xy'):
#     """    	
#     Returns coordinate matrices from coordinate vectors.
#     """
#     return _helper('meshgrid', *arrays, indexing=indexing)

def ones(impl, shape, dtype=None, device=None):
    """
    Returns a new array having a specified shape and filled with ones.
    """
    return _helper_from_shape('ones', impl, shape, dtype=dtype, device=device)

def ones_like(x, dtype=None, device=None):
    """
    Returns a new array filled with ones and having the same shape as an input array x.
    """
    return _helper_from_array('ones_like', x, dtype=dtype, device=device)

def tril(x, k=0):
    """
    Returns the lower triangular part of a matrix (or a stack of matrices) x.
    """
    return _helper_from_array('tril', x, k=k)

def triu(x, k=0):
    """
    Returns the upper triangular part of a matrix (or a stack of matrices) x.
    """
    return _helper_from_array('triu', x, k=k)

def zeros(impl, shape, dtype=None, device=None):
    """
    Returns a new array having a specified shape and filled with zeros.
    """
    return _helper_from_shape('zeros', impl, shape, dtype=dtype, device=device)

def zeros_like(x, dtype=None, device=None):
    """
    Returns a new array filled with zeros and having the same shape as an input array x.
    """
    return _helper_from_array('zeros_like', x, dtype=dtype, device=device)