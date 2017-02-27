# Copyright 2014-2017 The ODL development group
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
from odl.discr import DiscreteLp
from odl.space.base_ntuples import FnBase

import numpy as np

__all__ = ('SamplingOperator', 'ZerofillingOperator', 
           'VectoriseOperator', 'EmbeddingOperator')

class SamplingOperator(Operator):

    """Operator that samples coefficients.
    The operator is defined by 
        ``SamplingOperator(f, indices) == cell_volume * f[indices]``
    """

    def __init__(self, domain, indices, weighting=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `FnBase` or `DiscreteLp`
            Set of elements on which this operator acts.
        indices : list
            List of indices where to sample the function
        weighting : {None, 'cell_volume'}
            weighting scheme

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[0, 0], max_pt=[1, 1], shape=[2, 2])
        >>> x = X.element(range(X.size))
        >>> ind = [[0, 1], [1, 1]]
        >>> A = odl.SamplingOperator(X, ind)
        >>> A(x)
        rn(2).element([1.0, 3.0])
        >>> A(x)  # Out-of-place
        rn(2).element([1.0, 3.0])
        """
        if not isinstance(domain, (FnBase, DiscreteLp)):
            raise TypeError('`space` {!r} not a `FnBase` or `DiscreteLp` '
                            'instance'.format(domain))

        self.__indices = np.asarray(indices, dtype=int)
        # Converts a single sequence to an array of integers
        # and a sequence of arrays to a vertically stacked array
        if self.indices.ndim > 1:
            # Strides = increment in linear indices per axis
            strides = np.concatenate((np.cumprod(domain.shape[1:])[::-1], [1]))
            self.__indices_flat = np.sum(self.indices * strides[:, None], axis=0)
        else:
            self.__indices_flat = self.indices

        self.__weighting = weighting        
        if self.weighting is None:
            self.__weights = 1
        elif self.weighting == 'cell_volume':
            self.__weights = getattr(domain, 'cell_volume', 1.0)
        
        range = fn(self.indices_flat.size, dtype=domain.dtype)
        super().__init__(domain, range, linear=True)

    @property
    def weighting(self):
        """Weighting scheme for the sampling operator."""
        return self.__weighting

    @property
    def indices(self):
        """Indices where to sample the function."""
        return self.__indices

    @property
    def indices_flat(self):
        """Flat indices (linear indexing) where to sample the function."""
        return self.__indices_flat

    def _call(self, x, out=None):
        """Collect indices weighted with the cell volume."""
        if out is None:
            out = self.__weights * x[self.indices_flat]
        else:
            out[:] = self.__weights * x[self.indices_flat]
        return out

    @property
    def adjoint(self):
        """Adjoint of the sampling is 'zero filling'.

        Returns
        -------
        adjoint : `ZerofillingOperator`
        """
        if self.weighting is None:
            weighting = 'cell_volume'
        elif self.weighting == 'cell_volume':
            weighting = None
            
        return ZerofillingOperator(self.domain, self.indices, weighting)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self.domain, 
                                       self.indices)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

class ZerofillingOperator(Operator):

    """Adjoint of the sampling operator.
    If every index is sampled only once, then it performs a "zero filling" of
    of the sampled coefficients.

        ``ZerofillingOperator(f)(x) == sum_{y in I, y == x} f(y), 0 if x not in I``
    """

    def __init__(self, range, indices, weighting=None):
        """Initialize a new instance.

        Parameters
        ----------
        range : `FnBase` or `DiscreteLp`
            Set of elements into which this operator maps.
        indices : list
            List of indices (I in the formula above)
        weighting : {None, 'cell_volume'}
            weighting scheme

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[0, 0], max_pt=[1, 1], shape=[2, 2])
        >>> ind = [[0, 1], [1, 1]]
        >>> A = odl.ZerofillingOperator(X, ind)
        >>> x = A.domain.one()
        >>> A(x)
        uniform_discr([0.0, 0.0], [1.0, 1.0], (2, 2)).element([[0.0, 1.0], [0.0, 1.0]])
        >>> A(x)  # Out-of-place
        uniform_discr([0.0, 0.0], [1.0, 1.0], (2, 2)).element([[0.0, 1.0], [0.0, 1.0]])
        """

        self.__indices = np.asarray(indices, dtype=int)
        # Converts a single sequence to an array of integers
        # and a sequence of arrays to a vertically stacked array
        if self.indices.ndim > 1:
            # Strides = increment in linear indices per axis
            strides = np.concatenate((np.cumprod(range.shape[1:])[::-1], [1]))
            self.__indices_flat = np.sum(self.indices * strides[:, None], axis=0)
        else:
            self.__indices_flat = self.indices

        self.__weighting = weighting        
        if self.weighting is None:
            self.__weights = 1
        elif self.weighting == 'cell_volume':
            self.__weights = getattr(range, 'cell_volume', 1.0)

        domain = fn(self.indices_flat.size, dtype=range.dtype)
        super().__init__(domain, range, linear=True)
            
    @property
    def weighting(self):
        """Weighting scheme for the operator."""
        return self.__weighting

    @property
    def indices(self):
        """Indices where to sample the function."""
        return self.__indices

    @property
    def indices_flat(self):
        """Flattened set of indices (linear indexing) where to sample the function."""
        return self.__indices_flat

    def _call(self, x, out=None):
        """Sum all values if indices are given multiple times."""        
        y = np.bincount(self.indices_flat, weights=x, minlength=self.range.size)
            
        if out is None:
            out = self.__weights * y
        else:
            out[:] = self.__weights * y

        return out

    @property
    def adjoint(self):
        """Adjoint of the sampling is 'sampling'.

        Returns
        -------
        adjoint : `SamplingOperator`

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[4, 3])
        >>> x = X.element(range(1,X.size+1))
        >>> ind = [[0, 1, 2], [0, 1, 2]]
        >>> A = odl.SamplingOperator(X, ind)
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10
        True
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10 # Out-of-place
        True
        """
        if self.weighting is None:
            weighting = 'cell_volume'
        elif self.weighting == 'cell_volume':
            weighting = None
            
        return SamplingOperator(self.range, self.indices, weighting)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.indices)
                

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class VectoriseOperator(Operator):

    """Operator that reshapes the object as a column vector.

        ``VectoriseOperator(f) == np.ravel(f)``
    """

    def __init__(self, domain, scaling=1):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `FnBase` or `DiscreteLp`
            Set of elements on which this operator acts.

        Examples
        --------
        General usage example:

        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> x = X.element(range(X.size))
        >>> A = odl.VectoriseOperator(X)
        >>> A(x)
        rn(12).element([0.0, 1.0, 2.0, ..., 9.0, 10.0, 11.0])
        >>> A(x)  # Out-of-place
        rn(12).element([0.0, 1.0, 2.0, ..., 9.0, 10.0, 11.0])
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
            out = self.scaling * np.ravel(x)
        else:
            out[:] = self.scaling * np.ravel(x)
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
        adjoint : `EmbeddingOperator`

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[2, 3])
        >>> A = odl.VectoriseOperator(X)
        >>> x = A.domain.element(range(A.domain.size))
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10
        True
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10 # Out-of-place
        True
        """
        c = getattr(self.domain, 'cell_volume', 1.0) # TODO: This only works for a subset of function spaces
        return EmbeddingOperator(self.domain, self.scaling/c)

    @property
    def inverse(self):
        """Inverse of the vectorising is creating an element with these components.

        Returns
        -------
        adjoint : `EmbeddingOperator`

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> x = X.element(range(X.size))
        >>> A = odl.VectoriseOperator(X)
        >>> (A.inverse(A(x)) - x).norm() < 1e-10
        True
        >>> (A.inverse(A(x)) - x).norm() < 1e-10 # Out-of-place
        True
        """
        return EmbeddingOperator(self.domain, 1./self.scaling)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

class EmbeddingOperator(Operator):

    """Operator that creates an element of a vector space given its coefficients.
    The operator is defined by:

        ``EmbeddingOperator(X)(f) == X.element(f)``
        
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
        General usage example:
        
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[2, 2])
        >>> A = odl.EmbeddingOperator(X)
        >>> x = A.domain.element(range(A.domain.size))
        >>> A(x)
        uniform_discr([-1.0, -1.0], [1.0, 1.0], (2, 2)).element([[0.0, 1.0], [2.0, 3.0]])
        >>> A(x) # Out-of-place
        uniform_discr([-1.0, -1.0], [1.0, 1.0], (2, 2)).element([[0.0, 1.0], [2.0, 3.0]])
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
            out = self.scaling*x
        else:
            out[:] = self.scaling*x
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

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[2, 3])
        >>> A = odl.EmbeddingOperator(X)
        >>> x = A.domain.element(range(A.domain.size))
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10
        True
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10 # Out-of-place
        True
        """
        c = getattr(self.range, 'cell_volume', 1.0) # TODO: This only works for a subset of function spaces
        return VectoriseOperator(self.range, self.scaling*c)

    @property
    def inverse(self):
        """Inverse of the embedding is vectorising'.

        Returns
        -------
        adjoint : `VectoriseOperator`

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> A = odl.EmbeddingOperator(X)
        >>> x = A.domain.element(range(A.domain.size))
        >>> (A.inverse(A(x)) - x).norm() < 1e-10
        True
        >>> (A.inverse(A(x)) - x).norm() < 1e-10 # Out-of-place
        True
        """
        return VectoriseOperator(self.range, 1./self.scaling)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)
        
if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
    