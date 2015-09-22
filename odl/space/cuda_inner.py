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

"""Inner products with matrix weights, implemented in CUDA."""

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from future.utils import with_metaclass

# External
from abc import ABCMeta

# ODL
from odl.space.inner import WeightedInnerBase


__all__ = ('CudaConstantWeightedInner',)


class CudaWeightedInner(with_metaclass(ABCMeta, WeightedInnerBase)):

    """Abstract base class for CUDA weighted inner products. """

    def __call__(self, x, y):
        """`inner.__call__(x, y) <==> inner(x, y).`

        Calculate the inner product of `x` and `y` weighted by the
        matrix of this instance.

        Parameters
        ----------
        x, y : CudaVector
            Arrays whose inner product is calculated. They must have
            equal length.

        Returns
        -------
        inner : float or complex
            Weighted inner product. The output type depends on the
            input arrays and the weighting.
        """
        if x.size != y.size:
            raise TypeError('array sizes {} and {} are different.'
                            ''.format(x.size, y.size))

        # TODO: possibly adapt syntax once complex vectors are supported
        return self.matvec(x).inner(y)


class CudaMatrixWeightedInner(CudaWeightedInner):

    """Function object for matrix-weighted :math:`F^n` inner products.

    The weighted inner product with matrix :math:`G` is defined as

    :math:`<a, b> := b^H G a`

    with :math:`b^H` standing for transposed complex conjugate.
    """

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`."""
        raise NotImplementedError

    def matvec(self, vec):
        """Return product of the weighting matrix with a vector.

        Parameters
        ----------
        vec : CudaVector
            Array with which to multiply the weighting matrix

        Returns
        -------
        weighted : CudaVector
            The matrix-vector product as a CUDA vector
        """
        raise NotImplementedError


class CudaConstantWeightedInner(CudaWeightedInner):

    """Function object for constant-weighted :math:`F^n` inner products.

    The weighted inner product with constant :math:`c` is defined as

    :math:`<a, b> := b^H c a`

    with :math:`b^H` standing for transposed complex conjugate.
    """

    def __init__(self, constant):
        """Initialize a new instance.

        Parameters
        ----------
        constant : float
            Weighting constant of the inner product.
        """
        self.const = float(constant)

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`."""
        if other is self:
            return True
        elif isinstance(other, CudaConstantWeightedInner):
            return self.const == other.const
        elif isinstance(other, CudaWeightedInner):
            return other.__eq__(self)
        else:
            return False

    def matvec(self, vec):
        """Return product of the constant with a vector.

        Parameters
        ----------
        vec : CudaVector
            Array with which to multiply the constant

        Returns
        -------
        weighted : CudaVector
            The constant-vector product
        """
        newvec = type(vec)(vec.size)
        newvec.lincomb(self.const, vec, 0, vec)
        return newvec

    def __repr__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return 'CudaConstantWeightedInner({})'.format(self.const)

    def __str__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return '(x, y) --> {:.4} * y^H x'.format(self.const)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
