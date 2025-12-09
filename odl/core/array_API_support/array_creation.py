# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Array creation functions expected by the python array API.
Although ODL has many ways to create a tensor, we have found useful during development and testing to be able to create arrays in a certain backend.
We do not expect the users to work with these functions often but have still implemented them as we deemed useful during development.

Notes:
    -> the functions with name *_like take an array/ODL object as an input
    -> the other functions require impl, shape, dtype, device arguments.

Examples:
>>> odl.arange('numpy', 0,10,1, dtype='float32', device='cuda:0')
Traceback (most recent call last):
ValueError: Unsupported device for NumPy: 'cuda:0'
>>> odl.arange('numpy',start=0,stop=10,step=1, dtype='float32', device='cpu')
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.], dtype=float32)
>>> odl.asarray(odl.rn(4).element([1,2,3,4]))
array([ 1.,  2.,  3.,  4.])
>>> odl.full('numpy', (4,4), 4) == np.full((4,4),4)
array([[ True,  True,  True,  True],
       [ True,  True,  True,  True],
       [ True,  True,  True,  True],
       [ True,  True,  True,  True]], dtype=bool)
>>> odl.full_like(x = np.full((4,4),4), fill_value=4) == np.full((4,4),4)
array([[ True,  True,  True,  True],
       [ True,  True,  True,  True],
       [ True,  True,  True,  True],
       [ True,  True,  True,  True]], dtype=bool)
"""

from typing import Callable

from .utils import get_array_and_backend, lookup_array_backend

__all__ = (
    'arange',
    'asarray',
    'empty',
    'empty_like',
    'eye',
    # 'from_dlpack',
    'full',
    'full_like',
    'linspace',
    'meshgrid',
    'ones',
    'ones_like',
    'tril',
    'triu',
    'zeros',
    'zeros_like'
)


def _helper_from_impl(fname: str, impl: str, *args, **kwargs) -> Callable:
    """Internal helper to get function from impl string"""
    backend = lookup_array_backend(impl)
    fn = getattr(backend.array_namespace, fname)
    return fn(*args, **kwargs)


def _helper_from_array(fname: str, x, **kwargs) -> Callable:
    """Internal helper to get function from the backend infered from an array"""
    x, backend_x = get_array_and_backend(x)
    fn = getattr(backend_x.array_namespace, fname)
    return fn(x, **kwargs)

def arange(impl, start, stop=None, step=1, dtype=None, device=None):
    """
    Returns evenly spaced values within the half-open interval [start, stop) as a one-dimensional array.
    """
    return _helper_from_impl('arange', impl, start, stop=stop, step=step, dtype=dtype, device=device)

def asarray(x):
    """
    Returns an array corresponding to an ODL object.
    """
    return _helper_from_array('asarray', x)

def empty(impl, shape, dtype=None, device=None):
    """
    Returns an uninitialized array having a specified shape.
    """
    return _helper_from_impl('empty', impl, shape, dtype=dtype, device=device)

def empty_like(x, dtype=None, device=None):
    """
    Returns an uninitialized array with the same shape as an input array x.
    """
    return _helper_from_array('empty_like', x=x, dtype=dtype, device=device)

def eye(impl, n_rows, n_cols=None, k=0, dtype=None, device=None):
    """
    Returns a two-dimensional array with ones on the kth diagonal and zeros elsewhere.
    """
    return _helper_from_impl('eye', impl, n_rows=n_rows, n_cols=n_cols, k=k, dtype=dtype, device=device)

# def from_dlpack(x, device=None):
#     """
#     Returns a new array containing the data from another (array) object with a __dlpack__ method.
#     Note:
#         The device argument is currently NOT used, this is due to Pytorch needing to catch up with the array API standard
#     """
#     return _helper_from_array('from_dlpack', x=x)


def full(impl, shape, fill_value, dtype=None, device=None):
    """
    Returns a new array having a specified shape and filled with fill_value.
    """
    return _helper_from_impl('full', impl, shape=shape, fill_value=fill_value, dtype=dtype, device=device)

def full_like(x, fill_value, dtype=None, device=None):
    """
    Returns a new array filled with fill_value and having the same shape as an input array x.
    """
    return _helper_from_array('full_like', x=x, fill_value=fill_value, dtype=dtype, device=device)

def linspace(impl, start, stop, num, dtype=None, device=None, endpoint=True):
    """
    Returns evenly spaced numbers over a specified interval.
    """
    return _helper_from_impl('linspace', impl, start, stop, num, dtype=dtype, device=device, endpoint=endpoint)

def meshgrid(impl, *arrays, indexing='xy'):
    """    	
    Returns coordinate matrices from coordinate vectors.
    """
    return _helper_from_impl('meshgrid', impl, *arrays, indexing=indexing)

def ones(impl, shape, dtype=None, device=None):
    """
    Returns a new array having a specified shape and filled with ones.
    """
    return _helper_from_impl('ones', impl, shape=shape, dtype=dtype, device=device)

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
    return _helper_from_impl('zeros', impl, shape=shape, dtype=dtype, device=device)

def zeros_like(x, dtype=None, device=None):
    """
    Returns a new array filled with zeros and having the same shape as an input array x.
    """
    return _helper_from_array('zeros_like', x, dtype=dtype, device=device)