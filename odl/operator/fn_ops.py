## Copyright 2014-2016 The ODL development group
##
## This file is part of ODL.
##
## ODL is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ODL is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ODL.  If not, see <http://www.gnu.org/licenses/>.
#
#"""Default operators defined on any (reasonable) space."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.operator.operator import Operator
from odl.space import fn
from odl.set import LinearSpace, Field
#from odl.util import writable_array

import numpy as np

__all__ = ('SamplingOperator', 'ZerofillingOperator', 
           'VectoriseOperator', 'EmbeddingOperator')

class SamplingOperator(Operator):

    """Operator that samples coefficients.

        ``SamplingOperator(f)(i) == cell_volume * f(i) \approx \int_cell f(x) dx``
    """

    def __init__(self, domain, indices):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`
            Set of elements on which this operator acts.
        indices : list
            List of indices where to sample the function

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[0, 0], max_pt=[1, 1], shape=[2, 2])
        >>> x = X.element(range(X.size))
        >>> ind = list()
        >>> ind.append(range(2))
        >>> ind.append([1,1])
        >>> A = odl.operator.SamplingOperator(X, ind)
        >>> A(x)
        rn(2).element([0.25, 0.75])
        >>> A(x)  # Out-of-place
        rn(2).element([0.25, 0.75])
        """
        if not isinstance(domain, (LinearSpace, Field)):
            raise TypeError('`space` {!r} not a `LinearSpace` or `Field` '
                            'instance'.format(domain))

        self.__indices = np.asarray(indices, dtype=int)
        # Converts a single sequence to an array of integers
        # and a sequence of arrays to a vertically stacked array
        if self.indices.ndim > 1:
            # Strides = increment in linear indices per axis
            strides = np.concatenate((np.cumprod(domain.shape[1:])[::-1], [1]))
            self.__flat_indices = np.sum(self.indices * strides[:, None], axis=0)
        
        range = fn(self.flat_indices.size, dtype=domain.dtype)

        super().__init__(domain, range, linear=True)

    @property
    def indices(self):
        """indices where to sample the function."""
        return self.__indices

    @property
    def flat_indices(self):
        """indices where to sample the function."""
        return self.__flat_indices

    def _call(self, x, out=None):
        """Collect indices weighted with the cell volume."""
        c = getattr(self.domain, 'cell_volume', 1.0) # TODO: This only works for a subset of function spaces
        if out is None:
            out = x[self.flat_indices]*c
        else:
            out[:] = x[self.flat_indices]*c
        return out

    @property
    def adjoint(self):
        """Adjoint of the sampling is 'zero filling'.

        Returns
        -------
        adjoint : `ZerofillingOperator`
        """
        return ZerofillingOperator(self.domain, self.indices)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.indices)

    def __str__(self):
        """Return ``str(self)``."""
        return '{} * I'.format(self.indices)

class ZerofillingOperator(Operator):

    """Operator of multiplication with a scalar.

        ``ZerofillingOperator(f)(x) == sum_{y in I, y == x} f(y), 0 if x not in I``
    """

    def __init__(self, range, indices):
        """Initialize a new instance.

        Parameters
        ----------
        range : `LinearSpace` or `Field`
            Set of elements into which this operator maps.
        sample_indices : list
            List of indices (I in the formula above)

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[0, 0], max_pt=[1, 1], shape=[2, 2])
        >>> x = X.one()
        >>> ind = list()
        >>> ind.append(range(2))
        >>> ind.append([1,1])
        >>> A = odl.operator.SamplingOperator(X, ind)
        >>> A.adjoint(A(x))
        uniform_discr([0.0, 0.0], [1.0, 1.0], (2, 2)).element([[0.0, 0.25], [0.0, 0.25]])
        >>> A.adjoint(A(x))  # Out-of-place
        uniform_discr([0.0, 0.0], [1.0, 1.0], (2, 2)).element([[0.0, 0.25], [0.0, 0.25]])

        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[4, 3])
        >>> x = X.element(range(1,X.size+1))
        >>> ind = list()
        >>> ind.append(range(3))
        >>> ind.append(range(3))
        >>> A = odl.operator.SamplingOperator(X, ind)
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x))
        -1.7763568394002505e-15
        >>> A.adjoint(A(x))  # Out-of-place
        -1.7763568394002505e-15
        """

        self.__indices = np.asarray(indices, dtype=int)
        # Converts a single sequence to an array of integers
        # and a sequence of arrays to a vertically stacked array
        if self.indices.ndim > 1:
            # Strides = increment in linear indices per axis
            strides = np.concatenate((np.cumprod(range.shape[1:])[::-1], [1]))
            self.__flat_indices = np.sum(self.indices * strides[:, None], axis=0)

        self.__flat_indices_unique = np.unique(self.flat_indices)

        domain = fn(self.flat_indices.size, dtype=range.dtype)
        super().__init__(domain, range, linear=True)
            
    @property
    def indices(self):
        """indices where to sample the function."""
        return self.__indices

    @property
    def flat_indices(self):
        """flattened set of indices where to sample the function."""
        return self.__flat_indices

    @property
    def flat_indices_unique(self):
        """unique set of indices where to sample the function."""
        return self.__flat_indices_unique

    def _call(self, x, out=None):
        """sum all values if indices are given multiple times."""        
        y = np.bincount(self.flat_indices, weights=x, minlength=self.range.size)
            
        if out is None:
            out = self.range.element(y)
        else:
            out[:] = y

        return out

    @property
    def adjoint(self):
        """Adjoint of the sampling is 'sampling'.

        Returns
        -------
        adjoint : `SamplingOperator`
        """
        return SamplingOperator(self.domain, self.indices)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.indices)
                

    def __str__(self):
        """Return ``str(self)``."""
        return '{} * I'.format(self.indices)


class VectoriseOperator(Operator):

    """Operator that reshapes the object as a column vector.

        ``VectoriseOperator(f) == f[:]``
    """

    def __init__(self, domain, scaling=1):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`
            Set of elements on which this operator acts.

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> x = X.element(range(X.size))
        >>> A = odl.operator.VectoriseOperator(X)
        >>> A(x)
        rn(12).element([0.0, 1.0, 2.0, ..., 9.0, 10.0, 11.0])
        >>> A(x)  # Out-of-place
        rn(12).element([0.0, 1.0, 2.0, ..., 9.0, 10.0, 11.0])

        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> x = X.element(range(X.size))
        >>> A = odl.operator.VectoriseOperator(X)
        >>> (A.inverse(A(x)) - x).norm()
        0.0
        >>> (A.inverse(A(x)) - x).norm() # Out-of-place
        0.0
        """
        if not isinstance(domain, (LinearSpace, Field)):
            raise TypeError('`space` {!r} not a `LinearSpace` or `Field` '
                            'instance'.format(domain))
        
        self.__scaling = scaling        
        range = fn(domain.size, dtype=domain.dtype)
        super().__init__(domain, range, linear=True)

    def _call(self, x, out=None):
        """Collect indices."""
        if out is None:
            out = self.scaling*x[:]
        else:
            out[:] = self.scaling*x[:]
        return out

    @property
    def scaling(self):
        """scaling factor."""
        return self.__scaling
    
    @property
    def adjoint(self):
        """Adjoint of the vectorising is creating an element with these components
        with an appropriate weighting.

        Returns
        -------
        adjoint : `VectoriseOperatorAdjoint`
        """
        c = getattr(self.domain, 'cell_volume', 1.0) # TODO: This only works for a subset of function spaces
        return EmbeddingOperator(self.domain, self.scaling/c)

    @property
    def inverse(self):
        """Inverse of the vectorising is creating an element with these components.

        Returns
        -------
        adjoint : `EmbeddingOperator`
        """
        return EmbeddingOperator(self.domain, 1./self.scaling)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}'


class EmbeddingOperator(Operator):

    """Operator that creates an element of a vector space given its coefficients.

        ``EmbeddingOperator(f) == X.element(f)``
        
        For the adjoint it is important that the element is scaled properly.
    """

    def __init__(self, range, scaling=1):
        """Initialize a new instance.

        Parameters
        ----------
        range : `LinearSpace` or `Field`
            Space that the operator should map to.

        Examples
        --------
        Test for the adjoint
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> A = odl.operator.VectoriseOperator(X)
        >>> x = A.domain.element(range(A.domain.size))
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x))
        0.0
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) # Out-of-place
        0.0

        Test for the adjoint of the adjoint
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> A = odl.operator.VectoriseOperator(X).adjoint
        >>> x = A.domain.element(range(A.domain.size))
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x))
        0.0
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) # Out-of-place
        0.0

        Test for the adjoint of the inverse
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> A = odl.operator.VectoriseOperator(X).inverse
        >>> x = A.domain.element(range(A.domain.size))
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x))
        0.0
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) # Out-of-place
        0.0
        """
        if not isinstance(range, (LinearSpace, Field)):
            raise TypeError('`space` {!r} not a `LinearSpace` or `Field` '
                            'instance'.format(range))

        self.__scaling = scaling        
        domain = fn(range.size, dtype=range.dtype)
        super().__init__(domain, range, linear=True)

    def _call(self, x, out=None):
        """Create a new element with the known coefficients."""
        if out is None:
            out = self.range.element(self.scaling*x)
        else:
            out[:] = self.scaling*x[:]
        return out

    @property
    def scaling(self):
        """scaling factor."""
        return self.__scaling

    @property
    def adjoint(self):
        """Adjoint of the embedding is vectorising with an appropriate scaling'.

        Returns
        -------
        adjoint : `VectoriseOperator`
        """
        c = getattr(self.range, 'cell_volume', 1.0) # TODO: This only works for a subset of function spaces
        return VectoriseOperator(self.range, self.scaling/c)

    @property
    def inverse(self):
        """Inverse of the embedding is vectorising'.

        Returns
        -------
        adjoint : `VectoriseOperator`
        """
        return VectoriseOperator(self.range, 1./self.scaling)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}'

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()