# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Utility functions for space implementations."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

__all__ = ('vector',)

# External
import numpy as np

# Internal
from odl.space import CUDA_AVAILABLE
from odl.space.ntuples import Rn, Cn, Fn, Ntuples
from odl.util.utility import (
    is_real_floating_dtype, is_complex_floating_dtype, is_scalar_dtype)
if CUDA_AVAILABLE:
    from odl.space.cu_ntuples import CudaRn, CudaFn, CudaNtuples
    CudaCn = type(None)  # TODO: add when available
else:
    CudaRn = CudaCn = CudaFn = CudaNtuples = type(None)


def vector(array, dtype=None, impl='numpy'):
    """Create an n-tuples type vector from an array.

    Parameters
    ----------
    array : array-like
        Array from which to create the vector. All input is converted
        to 1d, i.e. scalars become 1-dim. vectors and multidimensional
        arrays are flattened (C-ordering).
    dtype : `object`, optional
        Set the data type of the vector manually with this option.
        By default, the space type is inferred from the input data.
    impl : {'numpy', 'cuda'}
        Implementation backend for the vector

    Returns
    -------
    vec : `NtuplesBaseVector`
        Vector created from the input array. Its concrete type depends
        on the provided arguments.

    Notes
    -----
    This is a convenience function and not intended for use in
    speed-critical algorithms. It creates a NumPy array first, hence
    especially CUDA vectors as input result in a large speed penalty.

    Examples
    --------
    >>> vector([1, 2, 3])  # No automatic cast to float
    Fn(3, 'int').element([1, 2, 3])
    >>> vector([1, 2, 3], dtype=float)
    Rn(3).element([1.0, 2.0, 3.0])
    >>> vector([1 + 1j, 2, 3 - 2j])
    Cn(3).element([(1+1j), (2+0j), (3-2j)])

    Non-scalar types are also supported:

    >>> vector([u'Hello,', u' world!'])
    Ntuples(2, '<U7').element([u'Hello,', u' world!'])

    Output is always a one-dimensional vector:

    >>> vector(0.0)
    Rn(1).element([0.0])
    >>> vector([[1, 0], [1, 1]])
    Fn(4, 'int').element([1, 0, 1, 1])
    """
    # TODO: disallow more than 1d?
    if dtype is None:
        # Cannot simply pass None since np.array interprets it as float
        arr = np.array(array, copy=False, ndmin=1).ravel()
    else:
        arr = np.array(array, copy=False, dtype=dtype, ndmin=1).ravel()

    if str(impl).lower() == 'numpy':
        if is_real_floating_dtype(arr.dtype):
            space_type = Rn
        elif is_complex_floating_dtype(arr.dtype):
            space_type = Cn
        elif is_scalar_dtype(arr.dtype):
            space_type = Fn
        else:
            space_type = Ntuples

    elif str(impl).lower() == 'cuda':
        if not CUDA_AVAILABLE:
            raise ValueError('CUDA implementation not available.')

        if is_real_floating_dtype(arr.dtype):
            space_type = CudaRn
        elif is_complex_floating_dtype(arr.dtype):
            space_type = CudaCn
            raise NotImplementedError('complex spaces in CUDA not supported.')
        elif is_scalar_dtype(arr.dtype):
            space_type = CudaFn
        else:
            space_type = CudaNtuples

    else:
        raise ValueError("implementation '{}' not understood.".format(impl))

    return space_type(len(arr), dtype=arr.dtype).element(arr)
