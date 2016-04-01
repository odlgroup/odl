# Copyright 2014-2016 The ODL development group
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

"""Operators defined for tensor fields."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super
from future.utils import raise_from

import numpy as np

from odl.operator.operator import Operator
from odl.set.sets import RealNumbers
from odl.space.fspace import FunctionSpace
from odl.space.pspace import ProductSpace


__all__ = ('PointwiseOperator', 'PointwiseNorm')


class PointwiseOperator(Operator):

    """Abstract operator for point-wise tensor field manipulations.

    A point-wise operator acts on a space of vector or tensor fields,
    i.e. a power space ``X^d`` of a discretized function space ``X``.
    Its range is the power space ``X^k`` with a possibly different
    number ``k`` of components.

    For example, if ``X`` is a `DiscreteLp` space, then
    ``ProductSpace(X, d)`` is a valid domain for any positive integer
    ``d``.

    Currently, only vector fields, i.e. one-dimensional products of
    ``X``, are supported.

    See also
    --------
    ProductSpace
    """

    def __init__(self, domain, ran_size, linear=False):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `ProductSpace` of `Discretization`
            Space of vector fields on which the operator acts.
            It has to be a product space of identical discretized
            function spaces, i.e. a power space.
        ran_size : positive `int`
            Number of components of the range
        linear : `bool`, optional
            If `True`, assume that the operator is linear.
        """
        if not isinstance(domain, ProductSpace):
            raise TypeError('domain {!r} is not a ProductSpace instance.'
                            ''.format(domain))
        if not domain.is_power_space:
            raise TypeError('domain {!r} is not a power space.'.format(domain))

        if not isinstance(domain[0].uspace, FunctionSpace):
            raise TypeError('domain component {!r} is not a function space '
                            'discretization.'.format(domain[0]))

        ran_size, ran_size_in = int(ran_size), ran_size
        if ran_size <= 0:
            raise ValueError('number of range components must be positive, '
                             'got {}.'.format(ran_size_in))
        self._ran_size = ran_size

        # Range will be just 'X' if there is only one component
        if self.ran_size == 1:
            ran = domain[0]
        else:
            ran = ProductSpace(domain[0], self.ran_size)
        super().__init__(domain=domain, range=ran, linear=linear)

    @property
    def discr_fspace(self):
        """The base space ``X`` of this operator's domain and range."""
        return self.domain[0]

    @property
    def ran_size(self):
        """Number of component spaces in the range."""
        return self._ran_size


class PointwiseNorm(PointwiseOperator):

    """Take the point-wise norm of a tensor field.

    This operator takes the (weighted) ``p``-norm

        ``||F(x)|| = [ sum_j( w_j * |F_j(x)|^p ) ]^(1/p)``

    for ``p`` finite and

        ``||F(x)|| = max_j( w_j * |F_j(x)| )``

    for ``p = inf``, where ``F`` is a tensor field. This implies that
    the `Operator.domain` is a power space of a discretized function
    space. For example, if ``X`` is a `DiscreteLp` space, then
    ``ProductSpace(X, d)`` is a valid domain for any positive integer
    ``d``.

    Currently only vector fields, i.e. product spaces with lenght-1
    shape, are supported.
    """

    def __init__(self, domain, exponent=None, weight=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `ProductSpace`
            Space of vector fields on which the operator acts.
            It has to be a product space of identical discretized
            function spaces, i.e. a power space. Its
            `ProductSpace.shape` must have length 1.
        exponent : non-zero `float`, optional
            Exponent of the norm in each point. Values between
            0 and 1 are currently not supported due to numerical
            instability.
            Default: ``domain.exponent``
        weight : `array-like` or `float`, optional
            Weighting array or constant for the norm. If an array is
            given, its length must be equal to ``domain.size``, and
            all entries must be positive. A provided constant must be
            positive.
            By default, the weights are is taken from
            ``domain.weighting``. Note that this excludes unusual
            weightings with custom inner product, norm or dist.
        """
        super().__init__(domain, ran_size=1, linear=False)
        if len(self.domain.shape) != 1:
            raise NotImplementedError

        if exponent is None:
            if domain.exponent is None:
                raise ValueError('cannot determine exponent from {}.'
                                 ''.format(self.domain))
            self._exponent = self.domain.exponent
        elif 0 <= exponent < 1:
            raise ValueError('exponent between 0 and 1 not allowed.')
        else:
            self._exponent = float(exponent)

        # Handle weighting, including sanity checks
        if weight is None:
            if hasattr(self.domain.weighting, 'vector'):
                self._weights = self.domain.weighting.vector
            elif hasattr(self.domain.weighting, 'const'):
                self._weights = (self.domain.weighting.const *
                                 np.ones(len(self.domain)))
            else:
                raise ValueError('weighting scheme {!r} of the domain does '
                                 'not define a weighting vector or constant.'
                                 ''.format(self.domain.weighting))
        elif np.isscalar(weight):
            if weight <= 0:
                raise ValueError('weighting constant must be positive, got '
                                 '{}.'.format(weight))
            self._weights = float(weight) * np.ones(self.domain.size)
        else:
            self._weights = np.asarray(weight, dtype='float64')
            if (not np.all(self.weights > 0) or
                    not np.all(np.isfinite(self.weights))):
                raise ValueError('weighting array {} contains invalid '
                                 'entries.'.format(weight))
        self._is_weighted = (not np.all(np.array_equiv(self.weights, 1.0)))

    @property
    def exponent(self):
        """Exponent ``p`` of this norm."""
        return self._exponent

    @property
    def weights(self):
        """Weighting vector of this norm."""
        return self._weights

    @property
    def is_weighted(self):
        """`True` if weighting is not 1 or all ones."""
        return self._is_weighted

    def _call(self, f, out):
        """Implement ``self(f, out)``."""
        if len(self.domain.shape) > 1:
            raise NotImplementedError

        if self.exponent == 1.0:
            self._call_vecfield_1(f, out)
        elif self.exponent == float('inf'):
            self._call_vecfield_inf(f, out)
        else:
            self._call_vecfield_p(f, out)

    def _call_vecfield_1(self, vf, out):
        """Implement ``self(vf, out)`` for exponent 1."""
        vf[0].ufunc.absolute(out=out)
        if self.is_weighted:
            out *= self.weights[0]

        if self.domain.size == 1:
            return

        tmp = self.range.element()
        for fi, w in zip(vf[1:], self.weights[1:]):
            fi.ufunc.absolute(out=tmp)
            if self.is_weighted:
                tmp *= w
            out += tmp

    def _call_vecfield_inf(self, vf, out):
        """Implement ``self(vf, out)`` for exponent ``inf``."""
        vf[0].ufunc.absolute(out=out)
        if self.is_weighted:
            out *= self.weights[0]

        if self.domain.size == 1:
            return

        tmp = self.range.element()
        for fi, w in zip(vf[1:], self.weights[1:]):
            fi.ufunc.absolute(out=tmp)
            if self.is_weighted:
                tmp *= w
            out.ufunc.maximum(tmp, out=out)

    def _call_vecfield_p(self, vf, out):
        """Implement ``self(vf, out)`` for exponent 1 < p < ``inf``."""
        # Optimization for 1 component - just absolute value (maybe weighted)
        if self.domain.size == 1:
            vf[0].ufunc.absolute(out=out)
            if self.is_weighted:
                out *= self.weights[0] ** (1 / self.exponent)
            return

        # Initialize out, avoiding one copy
        self._abs_pow_ufunc(vf[0], out=out)
        if self.is_weighted:
            out *= self.weights[0]

        tmp = self.range.element()
        for fi, w in zip(vf[1:], self.weights[1:]):
            self._abs_pow_ufunc(fi, out=tmp)
            if self.is_weighted:
                tmp *= w
            out += tmp

        out.ufunc.power(1 / self.exponent, out=out)

    def _abs_pow_ufunc(self, fi, out):
        """Compute |F_i(x)|^p point-wise and write to ``out``."""
        # Optimization for a very common case
        if self.exponent == 2.0 and self.discr_fspace.field == RealNumbers():
            out.multiply(fi, fi)
        else:
            fi.ufunc.absolute(out=out)
            out.ufunc.power(self.exponent, out=out)


class PointwiseInner(PointwiseOperator):

    """Take the point-wise norm of a vector field.

    This operator takes the (weighted) inner product

        ``<F(x), G(x)> = sum_j ( w_j * F_j(x) * conj(G_j(x)) )

    for a given vector field ``G``, where ``F`` is the vector field
    acting as a variable to this operator.

    This implies that the `Operator.domain` is a power space of a
    discretized function space. For example, if ``X`` is a `DiscreteLp`
    space, then ``ProductSpace(X, d)`` is a valid domain for any
    positive integer ``d``.
    """

    def __init__(self, domain, vecfield, weight=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `ProductSpace`
            Space of vector fields on which the operator acts.
            It has to be a product space of identical discretized
            function spaces, i.e. a power space.
        vecfield : domain `element-like`
            Vector field with which to calculate the point-wise inner
            product of an input vector field
        weight : `array-like` or `float`, optional
            Weighting array or constant for the norm. If an array is
            given, its length must be equal to ``domain.size``, and
            all entries must be positive. A provided constant must be
            positive.
            By default, the weights are is taken from
            ``domain.weighting``. Note that this excludes unusual
            weightings with custom inner product, norm or dist.
        """
        super().__init__(domain, ran_size=1, linear=False)
        if len(self.domain.shape) != 1:
            raise NotImplementedError

        if self.domain.field == RealNumbers():
            self._vecfield_conj = self.domain.element(vecfield)
        else:
            if not hasattr(self.discr_fspace.element_type, 'conj'):
                raise NotImplementedError(
                    'base space element type {!r} does not implement conj() '
                    'method required for complex inner products.'
                    ''.format(self.discr_fspace.element_type))
            try:
                self._vecfield_conj = self.domain.element(
                    [self.discr_fspace.element(vfi).conj()
                     for vfi in vecfield])
            except TypeError as err:
                raise_from(TypeError(
                    'unable to create an element of {!r} from {!r}.'
                    ''.format(self.domain, vecfield)), err)

        # Handle weighting, including sanity checks
        if weight is None:
            if hasattr(self.domain.weighting, 'vector'):
                self._weights = self.domain.weighting.vector
            elif hasattr(self.domain.weighting, 'const'):
                self._weights = (self.domain.weighting.const *
                                 np.ones(len(self.domain)))
            else:
                raise ValueError('weighting scheme {!r} of the domain does '
                                 'not define a weighting vector or constant.'
                                 ''.format(self.domain.weighting))
        elif np.isscalar(weight):
            if weight <= 0:
                raise ValueError('weighting constant must be positive, got '
                                 '{}.'.format(weight))
            self._weights = float(weight) * np.ones(self.domain.size)
        else:
            self._weights = np.asarray(weight, dtype='float64')
            if (not np.all(self.weights > 0) or
                    not np.all(np.isfinite(self.weights))):
                raise ValueError('weighting array {} contains invalid '
                                 'entries.'.format(weight))
        self._is_weighted = (not np.all(np.array_equiv(self.weights, 1.0)))

    def vecfield(self):
        """Fixed vector field ``G`` of this inner product.

        Internally, the complex conjugate of the vector field is
        stored for efficiency reasons, so for a complex domain, this
        method does not run in constant time.
        """
        if self.domain.field == RealNumbers():
            return self._vecfield_conj
        else:
            return self._vecfield_conj.conj()

    @property
    def weights(self):
        """Weighting vector of this norm."""
        return self._weights

    @property
    def is_weighted(self):
        """`True` if weighting is not 1 or all ones."""
        return self._is_weighted

    def _call(self, vf, out):
        """Implement ``self(vf, out)``."""
        out.multiply(vf[0], self._vecfield_conj[0])
        if self.is_weighted:
            out *= self.weights[0]

        if self.domain.size == 1:
            return

        tmp = self.range.element()
        for fi, gi_conj, w in zip(vf[1:], self._vecfield_conj[1:],
                                  self.weights[1:]):
            tmp.multiply(fi, gi_conj)
            if self.is_weighted:
                tmp *= w
            out += tmp


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
