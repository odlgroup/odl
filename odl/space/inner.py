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

"""Inner products with matrix weights, implemented with SciPy/NumPy."""

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from future.utils import with_metaclass

# External
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sp


__all__ = ('ConstantWeightedInner', 'MatrixWeightedInner')


class WeightedInnerBase(with_metaclass(ABCMeta, object)):

    """Abstract base class for weighted inner products.

    This class and its subclasses serve as a simple means to evaluate
    and compare weighted inner products semantically rather than by
    identity on a pure function level.
    """

    @abstractmethod
    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is a `WeightedInnerBase` instance
            represented by the same matrix, `False` otherwise.
        """

    @abstractmethod
    def matvec(self, vec):
        """Return product of the weighting matrix with a vector.

        Parameters
        ----------
        vec : array-like
            Array with which to multiply the weighting matrix

        Returns
        -------
        weighted : 1-dim. ndarray
            The matrix-vector product as a NumPy array
        """

    @abstractmethod
    def __call__(self, x, y):
        """`inner.__call__(x, y) <==> inner(x, y).`"""


class WeightedInner(with_metaclass(ABCMeta, WeightedInnerBase)):

    """Abstract base class for NumPy weighted inner products. """

    def __call__(self, x, y):
        """`inner.__call__(x, y) <==> inner(x, y).`

        Calculate the inner product of `x` and `y` weighted by the
        matrix of this instance.

        Parameters
        ----------
        x, y : array-like
            Arrays whose inner product is calculated. They must be
            one-dimensional and have equal length.

        Returns
        -------
        inner : scalar NumPy dtype
            Weighted inner product. The output type depends on the
            input arrays and the weighting.

        See also
        --------
        numpy.vdot : unweighted inner product
        """
        x, y = np.asarray(x), np.asarray(y)
        if not x.ndim == y.ndim == 1:
            raise TypeError('arrays {!r} and {!r} are not one-dimensional.'
                            ''.format(x, y))
        if x.shape != y.shape:
            raise TypeError('array shapes {} and {} are different.'
                            ''.format(x.shape, y.shape))

        # vdot conjugates the first argument if complex
        return np.vdot(y, self.matvec(x))


class MatrixWeightedInner(WeightedInner):

    """Function object for matrix-weighted :math:`F^n` inner products.

    The weighted inner product with matrix :math:`G` is defined as

    :math:`<a, b> := b^H G a`

    with :math:`b^H` standing for transposed complex conjugate.
    """

    def __init__(self, matrix):
        """Initialize a new instance.

        Parameters
        ----------
        matrix : array-like or scipy.sparse.spmatrix
            Weighting matrix of the inner product.
        """
        if isinstance(matrix, sp.sparse.spmatrix):
            self.matrix = matrix
            self.matrix_type = sp.sparse.spmatrix
        else:
            self.matrix = np.asmatrix(matrix)
            self.matrix_type = np.matrix
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError('matrix with shape {} is not square.'
                             ''.format(self.matrix.shape))

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`."""
        if other is self:
            return True
        elif isinstance(other, MatrixWeightedInner):
            if self.matrix_type == np.matrix:
                if other.matrix_type == np.matrix:
                    return np.array_equal(self.matrix, other.matrix)
                else:
                    return np.array_equal(self.matrix, other.matrix.todense())

            else:  # self.matrix_type == sp.sparse.spmatrix
                if other.matrix_type == np.matrix:
                    return np.array_equal(self.matrix.todense(), other.matrix)
                else:
                    return (self.matrix - other.matrix).nnz == 0
        elif isinstance(other, ConstantWeightedInner):
            if self.matrix_type == np.matrix:
                return np.array_equal(
                    self.matrix, np.eye(self.matrix.shape[0]) * other.const)
            else:
                # Construct sparse matrix with constant diagonal
                const_diag = np.empty(self.matrix.shape[0])
                const_diag.fill(other.const)
                const_mat = sp.sparse.dia_matrix((const_diag, [0]),
                                                 self.matrix.shape)
                return (self.matrix - const_mat).nnz == 0
        else:
            return False

    def matvec(self, vec):
        """Return product of the weighting matrix with a vector.

        Parameters
        ----------
        vec : array-like
            Array with which to multiply the weighting matrix

        Returns
        -------
        weighted : 1-dim. ndarray
            The matrix-vector product as a NumPy array
        """
        return np.asarray(self.matrix.dot(vec)).squeeze()

    def __repr__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        if self.matrix_type == sp.sparse.spmatrix:
            return ('MatrixWeightedInner(<{} sparse matrix, format {!r}, '
                    '{} stored entries>)'
                    ''.format(self.matrix.shape, self.matrix.format,
                              self.matrix.nnz))
        else:
            return 'MatrixWeightedInner(\n{!r}\n)'.format(self.matrix)

    def __str__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return '(x, y) --> y^H G x,  G =\n{}'.format(self.matrix)


class ConstantWeightedInner(WeightedInner):

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
        if isinstance(other, ConstantWeightedInner):
            return self.const == other.const
        elif isinstance(other, WeightedInner):
            return other.__eq__(self)
        else:
            return False

    def matvec(self, vec):
        """Return product of the constant with a vector.

        Parameters
        ----------
        vec : array-like
            Array with which to multiply the constant

        Returns
        -------
        weighted : 1-dim. ndarray
            The constant-vector product as a NumPy array
        """
        return np.asarray(self.const * vec).squeeze()

    def __repr__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return 'ConstantWeightedInner({})'.format(self.const)

    def __str__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return '(x, y) --> {:.4} * y^H x'.format(self.const)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
