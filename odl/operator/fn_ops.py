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

"""Default operators defined on fn (F^n where F is some field)."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super
import numpy as np

from odl.operator.operator import Operator
from odl.space import fn
from odl.set import LinearSpace, Field
from odl.discr import DiscreteLp
from odl.space.base_ntuples import FnBase

__all__ = ('SamplingOperator', 'WeightedSumSamplingOperator',
           'FlatteningOperator', 'FlatteningOperatorAdjoint')


class SamplingOperator(Operator):

    """Operator that samples coefficients.
    The operator is defined by::
        ``SamplingOperator(f) == c * f[indices]``
    with the weight c being determined by the variant. By choosing
    c = 1, this operator approximates point evaluations or inner products
    with dirac deltas, see option 'point_eval'. By choosing c = cell_volume
    it approximates the integration of f over the cell by multiplying its
    function valume with the cell volume, see option 'integrate'.
    """

    def __init__(self, domain, sampling_points, variant='point_eval'):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `FnBase` or `DiscreteLp`
            Set of elements on which this operator acts.
        sampling_points : sequence
            Sequence of indices that determine the sampling points.
            In n dimensions, it should be of length n. Each element of
            this list is a list by itself and should have the length of
            the total number of sampling points. Example: To sample a
            function at the points (0, 1) and (1, 1) the indices should
            be defined as sampling_points = [[0, 1], [1, 1]].
        variant : {'point_eval', 'integrate'}, optional
            For `'point_eval'` this operator performs the sampling by
            evaluation the function at the sampling points. The
            `'integrate'` variant approximates integration by
            multiplying point evaluation with the cell volume.

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[0, 0], max_pt=[1, 1], shape=[2, 2])
        >>> x = X.element(range(X.size))
        >>> sampling_points = [[0, 1], [1, 1]]
        >>> A = odl.SamplingOperator(X, sampling_points, 'point_eval')
        >>> A(x)
        rn(2).element([1.0, 3.0])
        >>> A = odl.SamplingOperator(X, sampling_points, 'integrate')
        >>> A(x)
        rn(2).element([0.25, 0.75])
        """
        if not isinstance(domain, (FnBase, DiscreteLp)):
            raise TypeError('`domain` {!r} not a `FnBase` or `DiscreteLp` '
                            'instance'.format(domain))

        self.__sampling_points = np.asarray(sampling_points, dtype=int)
        self.__variant = variant

        # Converts a single sequence to an array of integers
        # and a sequence of arrays to a vertically stacked array
        if self.sampling_points.ndim > 1:
            # Strides = increment in linear indices per axis
            strides = np.concatenate((np.cumprod(domain.shape[1:])[::-1], [1]))
            self.__indices_flat = np.sum(
                                    self.sampling_points * strides[:, None],
                                    axis=0)
        else:
            self.__indices_flat = self.sampling_points

        range = fn(self.indices_flat.size, dtype=domain.dtype)
        super().__init__(domain, range, linear=True)

    @property
    def variant(self):
        """Weighting scheme for the sampling operator."""
        return self.__variant

    @property
    def sampling_points(self):
        """Indices where to sample the function."""
        return self.__sampling_points

    @property
    def indices_flat(self):
        """Flat indices (linear indexing) where to sample the function."""
        return self.__indices_flat

    def _call(self, x, out=None):
        """Collect indices weighted with the cell volume."""
        if out is None:
            out = x[self.indices_flat]
        else:
            out[:] = x[self.indices_flat]

        if self.variant == 'point_eval':
            weights = 1.0
        elif self.variant == 'integrate':
            weights = getattr(self.domain, 'cell_volume', 1.0)
        else:
            raise RuntimeError('The variant "{!r}" is not yet supported'
                               ''.format(self.variant))

        if weights != 1.0:
            out *= weights

        return out

    @property
    def adjoint(self):
        """Adjoint of the sampling is summing over weighted dirac deltas or
        characteristic functions, see WeightedSumSamplingOperator.

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[4, 3])
        >>> x = X.element(range(1,X.size+1))
        >>> sampling_points = [[0, 1, 2], [0, 1, 2]]
        >>> A = odl.SamplingOperator(X, sampling_points, 'point_eval')
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10
        True
        >>> A = odl.SamplingOperator(X, sampling_points, 'integrate')
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10
        True
        """
        if self.variant == 'point_eval':
            variant = 'dirac'
        elif self.variant == 'integrate':
            variant = 'char_fun'
        else:
            raise RuntimeError('The variant "{!r}" is not yet supported'
                               ''.format(self.variant))

        return WeightedSumSamplingOperator(self.domain, self.sampling_points,
                                           variant)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self.domain,
                                       self.sampling_points)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class WeightedSumSamplingOperator(Operator):

    """Adjoint of the sampling operator.
    It sums dirac deltas or charactistic functions that are centered
    around the sampling points and weighted by its argument.

    .. math::
        W(g)(x) == \sum_{i in sampling_points} d_i(x) g_i
    """

    def __init__(self, range, sampling_points, variant='dirac'):
        """Initialize a new instance.

        Parameters
        ----------
        range : `FnBase` or `DiscreteLp`
            Set of elements into which this operator maps.
        sampling_points : sequence
            Sequence of indices that determine the sampling points.
            In n dimensions, it should be of length n. Each element of
            this list is a list by itself and should have the length of
            the total number of sampling points. Example: To sample a
            function at the points (0, 1) and (1, 1) the indices should
            be defined as sampling_points = [[0, 1], [1, 1]].
        variant : {'dirac', 'char_fun'}, optional
            This option determines which function to sum over.

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[0, 0], max_pt=[1, 1], shape=[2, 2])
        >>> sampling_points = [[0, 1], [1, 1]]
        >>> A = odl.WeightedSumSamplingOperator(X, sampling_points, 'dirac')
        >>> x = A.domain.one()
        >>> A(x)
        uniform_discr([0.0, 0.0], [1.0, 1.0], (2, 2)).element([[0.0, 0.25],
        [0.0, 0.25]])
        >>> A = odl.WeightedSumSamplingOperator(X, sampling_points, 'char_fun')
        >>> A(x)
        uniform_discr([0.0, 0.0], [1.0, 1.0], (2, 2)).element([[0.0, 1.0],
        [0.0, 1.0]])
        """

        self.__sampling_points = np.asarray(sampling_points, dtype=int)
        self.__variant = variant

        # Converts a single sequence to an array of integers
        # and a sequence of arrays to a vertically stacked array
        if self.sampling_points.ndim > 1:
            # Strides = increment in linear indices per axis
            strides = np.concatenate((np.cumprod(range.shape[1:])[::-1], [1]))
            self.__indices_flat = np.sum(
                                    self.sampling_points * strides[:, None],
                                    axis=0)
        else:
            self.__indices_flat = self.sampling_points

        domain = fn(self.indices_flat.size, dtype=range.dtype)
        super().__init__(domain, range, linear=True)

    @property
    def variant(self):
        """Weighting scheme for the operator."""
        return self.__variant

    @property
    def sampling_points(self):
        """Indices where to sample the function."""
        return self.__sampling_points

    @property
    def indices_flat(self):
        """Flattened set of indices (linear indexing) where to sample the
        function."""
        return self.__indices_flat

    def _call(self, x, out=None):
        """Sum all values if indices are given multiple times."""
        y = np.bincount(self.indices_flat, weights=x,
                        minlength=self.range.size)

        if out is None:
            out = y
        else:
            out[:] = y

        if self.variant == 'dirac':
            weights = getattr(self.range, 'cell_volume', 1.0)
        elif self.variant == 'char_fun':
            weights = 1.0
        else:
            raise RuntimeError('The variant "{!r}" is not yet supported'
                               ''.format(self.variant))

        if weights != 1.0:
            out *= weights

        return out

    @property
    def adjoint(self):
        """Adjoint of the weighted sum of sampling functions is 'sampling',
        see SamplingOperator.

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[4, 3])
        >>> sampling_points = [[0, 1, 2], [0, 1, 2]]
        >>> A = odl.WeightedSumSamplingOperator(X, sampling_points, 'dirac')
        >>> x = A.domain.one()
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10
        True
        >>> A = odl.WeightedSumSamplingOperator(X, sampling_points, 'char_fun')
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10
        True
        """
        if self.variant == 'dirac':
            variant = 'point_eval'
        elif self.variant == 'char_fun':
            variant = 'integrate'
        else:
            raise RuntimeError('The variant "{!r}" is not yet supported'
                               ''.format(self.variant))

        return SamplingOperator(self.range, self.sampling_points, variant)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.sampling_points)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class FlatteningOperator(Operator):

    """Operator that reshapes the object as a column vector.

        ``FlatteningOperator(f) == np.ravel(f)``
    """

    def __init__(self, domain, order='C'):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `FnBase` or `DiscreteLp`
            Set of elements on which this operator acts.
        order : {'C', 'F'} (optional)
            The flattening is performed in this order. 'C' means that
            that the last index is changing fastest or in terms of a matrix
            that the read out is row-by-row. Likewise 'F' is column-by-column.

        Examples
        --------
        General usage example:

        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[2, 3])
        >>> x = X.element(range(X.size))
        >>> A = odl.FlatteningOperator(X)
        >>> A(x)
        rn(6).element([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        """
        if not isinstance(domain, (LinearSpace, Field)):
            raise TypeError('`domain` {!r} not a `LinearSpace` or `Field` '
                            'instance'.format(domain))

        self.__order = order
        range = fn(domain.size, dtype=domain.dtype)
        super().__init__(domain, range, linear=True)

    def _call(self, x, out=None):
        """Collect indices"""
        if out is None:
            out = np.ravel(x, order=self.order)
        else:
            out[:] = np.ravel(x, order=self.order)
        return out

    @property
    def order(self):
        """order of the flattening"""
        return self.__order

    @property
    def adjoint(self):
        """Adjoint of the vectorising is creating an element with these
        components with an appropriate weighting, see
        FlatteningOperatorAdjoint.

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[2, 3])
        >>> A = odl.FlatteningOperator(X)
        >>> x = A.domain.element(range(A.domain.size))
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10
        True
        """
        # TODO: This only works for a subset of function spaces
        c = getattr(self.domain, 'cell_volume', 1.0)
        return 1./c * FlatteningOperatorAdjoint(self.domain)

    @property
    def inverse(self):
        """Reshaping to original shape, see FlatteningOperatorAdjoint.

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> x = X.element(range(X.size))
        >>> A = odl.FlatteningOperator(X)
        >>> (A.inverse(A(x)) - x).norm() < 1e-10
        True
        """
        return FlatteningOperatorAdjoint(self.domain)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class FlatteningOperatorAdjoint(Operator):

    """Operator that creates an element of a vector space given its coefficients:

        ``FlatteningOperatorAdjoint(f) == range.element(f)``
    """

    # TODO: give the domain as an optional argument
    def __init__(self, range):
        """Initialize a new instance.

        Parameters
        ----------
        range : `LinearSpace` or `Field`
            Space that the operator should map to.

        Examples
        --------
        General usage example:

        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[1, 2])
        >>> A = odl.FlatteningOperatorAdjoint(X)
        >>> x = A.domain.element(range(A.domain.size))
        >>> A(x)
        uniform_discr([-1.0, -1.0], [1.0, 1.0], (1, 2)).element([[0.0, 1.0]])
        """
        if not isinstance(range, (LinearSpace, Field)):
            raise TypeError('`range` {!r} not a `LinearSpace` or `Field` '
                            'instance'.format(range))

        domain = fn(range.size, dtype=range.dtype)
        super().__init__(domain, range, linear=True)

    def _call(self, x, out=None):
        """Create a new element with the known coefficients."""
        if out is None:
            out = x
        else:
            out[:] = x

        return out

    @property
    def adjoint(self):
        """Adjoint of this operation is flattening with an appropriate scaling,
        see FlatteningOperator.

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[2, 3])
        >>> A = odl.FlatteningOperatorAdjoint(X)
        >>> x = A.domain.element(range(A.domain.size))
        >>> A.adjoint(A(x)).inner(x) - A(x).inner(A(x)) < 1e-10
        True
        """
        # TODO: This only works for a subset of function spaces
        c = getattr(self.range, 'cell_volume', 1.0)
        return c * FlatteningOperator(self.range)

    @property
    def inverse(self):
        """Inverse of the embedding is flattening with the right scaling,
        see FlatteningOperator.

        Examples
        --------
        >>> X = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=[3, 4])
        >>> A = odl.FlatteningOperatorAdjoint(X)
        >>> x = A.domain.element(range(A.domain.size))
        >>> (A.inverse(A(x)) - x).norm() < 1e-10
        True
        """
        return FlatteningOperator(self.range)

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
