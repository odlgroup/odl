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
from odl.space.ntuples import Rn, Cn, Fn, Ntuples
from odl.util.utility import (
    is_real_floating_dtype, is_complex_floating_dtype, is_scalar_dtype)
from odl.space.cu_ntuples import CUDA_AVAILABLE, CudaRn, CudaFn, CudaNtuples


def vector(array, dtype=None, impl='numpy'):
    """Create an n-tuples type vector from an array.

    Parameters
    ----------
    array : array-like
        Array from which to create the vector. Scalars become
        one-dimensional vectors.
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

    Scalars become a one-element vector:

    >>> vector(0.0)
    Rn(1).element([0.0])
    """
    # Sanitize input
    arr = np.array(array, copy=False, ndmin=1)
    impl = str(impl).lower()

    # Validate input
    if arr.ndim > 1:
        raise ValueError('array has {} dimensions, expected 1.'
                         ''.format(arr.ndim))

    # Set dtype
    if dtype is not None:
        space_dtype = dtype
    elif arr.dtype == float and impl == 'cuda':
        # Special case, default float is float32 on cuda
        space_dtype = 'float32'
    else:
        space_dtype = arr.dtype

    # Select implementation
    if impl == 'numpy':
        if is_real_floating_dtype(space_dtype):
            space_type = Rn
        elif is_complex_floating_dtype(space_dtype):
            space_type = Cn
        elif is_scalar_dtype(space_dtype):
            space_type = Fn
        else:
            space_type = Ntuples

    elif impl == 'cuda':
        if not CUDA_AVAILABLE:
            raise ValueError('CUDA implementation not available.')

        if is_real_floating_dtype(space_dtype):
            space_type = CudaRn
        elif is_complex_floating_dtype(space_dtype):
            raise NotImplementedError('complex spaces in CUDA not supported.')
        elif is_scalar_dtype(space_dtype):
            space_type = CudaFn
        else:
            space_type = CudaNtuples

    else:
        raise ValueError("implementation '{}' not understood.".format(impl))

    return space_type(len(arr), dtype=space_dtype).element(arr)
