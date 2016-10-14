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

import numpy as np

from odl.operator import Operator
from odl.set import RealNumbers, ComplexNumbers, LinearSpace
from odl.space import ProductSpace


__all__ = ('PointwiseNorm', 'PointwiseInner')

_SUPPORTED_DIFF_METHODS = ('central', 'forward', 'backward')


class PointwiseTensorFieldOperator(Operator):

    """Abstract operator for point-wise tensor field manipulations.

    A point-wise operator acts on a space of vector or tensor fields,
    i.e. a power space ``X^d`` of a discretized function space ``X``.
    Its range is the power space ``X^k`` with a possibly different
    number ``k`` of components. For ``k == 1``, the base space
    ``X`` can be used instead.

    For example, if ``X`` is a `DiscreteLp` space, then
    ``ProductSpace(X, d)`` is a valid domain for any positive integer
    ``d``.

    Currently, only vector fields, i.e. one-dimensional products of
    ``X``, are supported.

    See Also
    --------
    ProductSpace
    """

    def __init__(self, domain, range, linear=False):
        """Initialize a new instance.

        Parameters
        ----------
        domain, range : {`ProductSpace`, `LinearSpace`}
            Spaces of vector fields between which the operator maps.
            They have to be either power spaces of the same base space
            ``X`` or the base space itself (only one of them).
            Empty product spaces are not allowed.
        linear : bool, optional
            If ``True``, assume that the operator is linear.
        """
        if isinstance(domain, ProductSpace):
            if not domain.is_power_space:
                raise TypeError('`domain` {!r} is not a power space'
                                ''.format(domain))

            if domain.size == 0:
                raise ValueError('`domain` is a product space of size 0')

            dom_base = domain[0]
        elif isinstance(domain, LinearSpace):
            dom_base = domain
        else:
            raise TypeError('`domain` {!r} is not a ProductSpace or '
                            'LinearSpace instance'.format(domain))

        if isinstance(range, ProductSpace):
            if not range.is_power_space:
                raise TypeError('`range` {!r} is not a power space'
                                ''.format(range))

            if range.size == 0:
                raise ValueError('`range` is a product space of size 0')

            ran_base = range[0]
        elif isinstance(range, LinearSpace):
            ran_base = range
        else:
            raise TypeError('`range` {!r} is not a ProductSpace or '
                            'LinearSpace instance'.format(range))

        if dom_base != ran_base:
            raise ValueError('`domain` and `range` have different base spaces '
                             '({!r} != {!r})'
                             ''.format(dom_base, ran_base))

        super().__init__(domain=domain, range=range, linear=linear)
        self.__base_space = dom_base

    @property
    def base_space(self):
        """Base space ``X`` of this operator's domain and range."""
        return self.__base_space


class PointwiseNorm(PointwiseTensorFieldOperator):

    """Take the point-wise norm of a vector field.

    This operator takes the (weighted) ``p``-norm

        ``||F(x)|| = [ sum_j( w_j * |F_j(x)|^p ) ]^(1/p)``

    for ``p`` finite and

        ``||F(x)|| = max_j( w_j * |F_j(x)| )``

    for ``p = inf``, where ``F`` is a vector field. This implies that
    the `Operator.domain` is a power space of a discretized function
    space. For example, if ``X`` is a `DiscreteLp` space, then
    ``ProductSpace(X, d)`` is a valid domain for any positive integer
    ``d``.
    """

    def __init__(self, vfspace, exponent=None, weight=None):
        """Initialize a new instance.

        Parameters
        ----------
        vfspace : `ProductSpace`
            Space of vector fields on which the operator acts.
            It has to be a product space of identical spaces, i.e. a
            power space.
        exponent : non-zero float, optional
            Exponent of the norm in each point. Values between
            0 and 1 are currently not supported due to numerical
            instability.
            Default: ``vfspace.exponent``
        weight : `array-like` or float, optional
            Weighting array or constant for the norm. If an array is
            given, its length must be equal to ``domain.size``, and
            all entries must be positive. A provided constant must be
            positive.
            By default, the weights are is taken from
            ``domain.weighting``. Note that this excludes unusual
            weightings with custom inner product, norm or dist.

        Examples
        --------
        We make a tiny vector field space in 2D and create the
        standard point-wise norm operator on that space. The operator
        maps a vector field to a scalar function:

        >>> spc = odl.uniform_discr([-1, -1], [1, 1], (1, 2))
        >>> vfspace = odl.ProductSpace(spc, 2)
        >>> pw_norm = PointwiseNorm(vfspace)
        >>> pw_norm.range == spc
        True

        Now we can calculate the 2-norm in each point:

        >>> x = vfspace.element([[[1, -4]],
        ...                      [[0, 3]]])
        >>> print(pw_norm(x))
        [[1.0, 5.0]]

        We can change the exponent either in the vector field space
        or in the operator directly:

        >>> vfspace = odl.ProductSpace(spc, 2, exponent=1)
        >>> pw_norm = PointwiseNorm(vfspace)
        >>> print(pw_norm(x))
        [[1.0, 7.0]]
        >>> vfspace = odl.ProductSpace(spc, 2)
        >>> pw_norm = PointwiseNorm(vfspace, exponent=1)
        >>> print(pw_norm(x))
        [[1.0, 7.0]]
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`vfspace` {!r} is not a ProductSpace '
                            'instance'.format(vfspace))
        super().__init__(domain=vfspace, range=vfspace[0], linear=False)

        # Need to check for product space shape once higher order tensors
        # are implemented

        if exponent is None:
            if self.domain.exponent is None:
                raise ValueError('cannot determine `exponent` from {}'
                                 ''.format(self.domain))
            self._exponent = self.domain.exponent
        elif exponent < 1:
            raise ValueError('`exponent` smaller than 1 not allowed')
        else:
            self._exponent = float(exponent)

        # Handle weighting, including sanity checks
        if weight is None:
            # TODO: find a more robust way of getting the weighs as a vector
            if hasattr(self.domain.weighting, 'vector'):
                self._weights = self.domain.weighting.vector
            elif hasattr(self.domain.weighting, 'const'):
                self._weights = (self.domain.weighting.const *
                                 np.ones(len(self.domain)))
            else:
                raise ValueError('weighting scheme {!r} of the domain does '
                                 'not define a weighting vector or constant'
                                 ''.format(self.domain.weighting))
        elif np.isscalar(weight):
            if weight <= 0:
                raise ValueError('weighting constant must be positive, got '
                                 '{}'.format(weight))
            self._weights = float(weight) * np.ones(self.domain.size)
        else:
            self._weights = np.asarray(weight, dtype='float64')
            if (not np.all(self.weights > 0) or
                    not np.all(np.isfinite(self.weights))):
                raise ValueError('weighting array {} contains invalid '
                                 'entries'.format(weight))
        self._is_weighted = not np.array_equiv(self.weights, 1.0)

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
        """``True`` if weighting is not 1 or all ones."""
        return self._is_weighted

    def _call(self, f, out):
        """Implement ``self(f, out)``."""
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
        for fi, wi in zip(vf[1:], self.weights[1:]):
            fi.ufunc.absolute(out=tmp)
            if self.is_weighted:
                tmp *= wi
            out += tmp

    def _call_vecfield_inf(self, vf, out):
        """Implement ``self(vf, out)`` for exponent ``inf``."""
        vf[0].ufunc.absolute(out=out)
        if self.is_weighted:
            out *= self.weights[0]

        if self.domain.size == 1:
            return

        tmp = self.range.element()
        for vfi, wi in zip(vf[1:], self.weights[1:]):
            vfi.ufunc.absolute(out=tmp)
            if self.is_weighted:
                tmp *= wi
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
        for fi, wi in zip(vf[1:], self.weights[1:]):
            self._abs_pow_ufunc(fi, out=tmp)
            if self.is_weighted:
                tmp *= wi
            out += tmp

        out.ufunc.power(1 / self.exponent, out=out)

    def _abs_pow_ufunc(self, fi, out):
        """Compute |F_i(x)|^p point-wise and write to ``out``."""
        # Optimization for a very common case
        if self.exponent == 2.0 and self.base_space.field == RealNumbers():
            fi.multiply(fi, out=out)
        else:
            fi.ufunc.absolute(out=out)
            out.ufunc.power(self.exponent, out=out)

    def derivative(self, vf):
        """Derivative of the point-wise norm operator at ``vf``.

        The derivative at ``F`` of the point-wise norm operator ``N``
        with finite exponent ``p`` and weights ``w`` is the pointwise
        inner product with the vector field

            ``x --> N(F)(x)^(1-p) * [ F_j(x) * |F_j(x)|^(p-2) ]_j``.

        Note that this is not well-defined for ``F = 0``. If ``p < 2``,
        any zero component will result in a singularity.

        Parameters
        ----------
        vf : `domain` `element-like`
            Vector field ``F`` at which to evaluate the derivative.

        Returns
        -------
        deriv : `PointwiseInner`
            Derivative operator at the given point ``vf``.

        Raises
        ------
        NotImplementedError
            * if the vector field space is complex, since the derivative
              is not linear in that case
            * if the exponent is ``inf``
        """
        if self.domain.field == ComplexNumbers():
            raise NotImplementedError('operator not Frechet-differentiable '
                                      'on a complex space')

        if self.exponent == float('inf'):
            raise NotImplementedError('operator not Frechet-differentiable '
                                      'for exponent = inf')

        vf = self.domain.element(vf)
        vf_pwnorm_fac = self(vf)
        vf_pwnorm_fac **= (self.exponent - 1)

        inner_vf = vf.copy()

        for gi in inner_vf:
            gi /= vf_pwnorm_fac * gi ** (self.exponent - 2)

        return PointwiseInner(self.domain, inner_vf, weight=self.weights)


class PointwiseInnerBase(PointwiseTensorFieldOperator):
    """Base class for `PointwiseInner` and `PointwiseInnerAdjoint`.

    Implemented to allow code reuse between the classes.
    """

    def __init__(self, adjoint, vfspace, vecfield, weight=None):
        """Initialize a new instance.

        All parameters are given according to the specifics of the "usual"
        operator. The ``adjoint`` parameter is used to control conversions
        for the inverse transform.

        Parameters
        ----------
        adjoint : bool
            ``True`` if the operator should be the adjoint, ``False``
            otherwise.
        vfspace : `ProductSpace`
            Space of vector fields on which the operator acts.
            It has to be a product space of identical spaces, i.e. a
            power space.
        vecfield : ``vfspace`` `element-like`
            Vector field with which to calculate the point-wise inner
            product of an input vector field
        weight : `array-like` or float, optional
            Weighting array or constant for the norm. If an array is
            given, its length must be equal to ``domain.size``, and
            all entries must be positive. A provided constant must be
            positive.
            By default, the weights are is taken from
            ``domain.weighting``. Note that this excludes unusual
            weightings with custom inner product, norm or dist.
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`vfsoace` {!r} is not a ProductSpace '
                            'instance'.format(vfspace))
        if adjoint:
            super().__init__(domain=vfspace[0], range=vfspace, linear=True)
        else:
            super().__init__(domain=vfspace, range=vfspace[0], linear=True)

        # Bail out if the space is complex but we cannot take the complex
        # conjugate.
        if (vfspace.field == ComplexNumbers() and
                not hasattr(self.base_space.element_type, 'conj')):
            raise NotImplementedError(
                'base space element type {!r} does not implement conj() '
                'method required for complex inner products'
                ''.format(self.base_space.element_type))

        self._vecfield = vfspace.element(vecfield)

        # Handle weighting, including sanity checks
        if weight is None:
            if hasattr(vfspace.weighting, 'vector'):
                self._weights = vfspace.weighting.vector
            elif hasattr(vfspace.weighting, 'const'):
                self._weights = (vfspace.weighting.const *
                                 np.ones(len(vfspace)))
            else:
                raise ValueError('weighting scheme {!r} of the domain does '
                                 'not define a weighting vector or constant'
                                 ''.format(vfspace.weighting))
        elif np.isscalar(weight):
            if weight <= 0:
                raise ValueError('weighting constant must be positive, got '
                                 '{}'.format(weight))
            self._weights = float(weight) * np.ones(vfspace.size)
        else:
            self._weights = np.asarray(weight, dtype='float64')
            if (not np.all(self.weights > 0) or
                    not np.all(np.isfinite(self.weights))):
                raise ValueError('weighting array {} contains invalid '
                                 'entries'.format(weight))
        self._is_weighted = not np.array_equiv(self.weights, 1.0)

    @property
    def vecfield(self):
        """Fixed vector field ``G`` of this inner product."""
        return self._vecfield

    @property
    def weights(self):
        """Weighting vector of this norm."""
        return self._weights

    @property
    def is_weighted(self):
        """``True`` if weighting is not 1 or all ones."""
        return self._is_weighted

    @property
    def adjoint(self):
        """Adjoint operator."""
        raise NotImplementedError('abstract method')


class PointwiseInner(PointwiseInnerBase):

    """Take the point-wise inner product with a given vector field.

    This operator takes the (weighted) inner product

        ``<F(x), G(x)> = sum_j ( w_j * F_j(x) * conj(G_j(x)) )``

    for a given vector field ``G``, where ``F`` is the vector field
    acting as a variable to this operator.

    This implies that the `Operator.domain` is a power space of a
    discretized function space. For example, if ``X`` is a `DiscreteLp`
    space, then ``ProductSpace(X, d)`` is a valid domain for any
    positive integer ``d``.
    """

    def __init__(self, vfspace, vecfield, weight=None):
        """Initialize a new instance.

        Parameters
        ----------
        vfspace : `ProductSpace`
            Space of vector fields on which the operator acts.
            It has to be a product space of identical spaces, i.e. a
            power space.
        vecfield : ``vfspace`` `element-like`
            Vector field with which to calculate the point-wise inner
            product of an input vector field
        weight : `array-like` or float, optional
            Weighting array or constant for the norm. If an array is
            given, its length must be equal to ``domain.size``, and
            all entries must be positive. A provided constant must be
            positive.
            By default, the weights are is taken from
            ``domain.weighting``. Note that this excludes unusual
            weightings with custom inner product, norm or dist.

        Examples
        --------
        We make a tiny vector field space in 2D and create the
        point-wise inner product operator with a fixed vector field.
        The operator maps a vector field to a scalar function:

        >>> spc = odl.uniform_discr([-1, -1], [1, 1], (1, 2))
        >>> vfspace = odl.ProductSpace(spc, 2)
        >>> fixed_vf = np.array([[[0, 1]],
        ...                      [[1, -1]]])
        >>> pw_inner = PointwiseInner(vfspace, fixed_vf)
        >>> pw_inner.range == spc
        True

        Now we can calculate the inner product in each point:

        >>> x = vfspace.element([[[1, -4]],
        ...                      [[0, 3]]])
        >>> print(pw_inner(x))
        [[0.0, -7.0]]
        """
        super().__init__(adjoint=False, vfspace=vfspace, vecfield=vecfield,
                         weight=weight)

    @property
    def vecfield(self):
        """Fixed vector field ``G`` of this inner product."""
        return self._vecfield

    @property
    def weights(self):
        """Weighting vector of this norm."""
        return self._weights

    @property
    def is_weighted(self):
        """``True`` if weighting is not 1 or all ones."""
        return self._is_weighted

    def _call(self, vf, out):
        """Implement ``self(vf, out)``."""
        if self.domain.field == ComplexNumbers():
            vf[0].multiply(self._vecfield[0].conj(), out=out)
        else:
            vf[0].multiply(self._vecfield[0], out=out)

        if self.is_weighted:
            out *= self.weights[0]

        if self.domain.size == 1:
            return

        tmp = self.range.element()
        for vfi, gi, wi in zip(vf[1:], self.vecfield[1:],
                               self.weights[1:]):

            if self.domain.field == ComplexNumbers():
                vfi.multiply(gi.conj(), out=tmp)
            else:
                vfi.multiply(gi, out=tmp)

            if self.is_weighted:
                tmp *= wi
            out += tmp

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `PointwiseInnerAdjoint`
        """
        return PointwiseInnerAdjoint(
            sspace=self.base_space, vecfield=self.vecfield,
            vfspace=self.domain, weight=self.weights)


class PointwiseInnerAdjoint(PointwiseInnerBase):

    """Adjoint of the point-wise inner product operator.

    The adjoint of the inner product operator is a mapping

        ``A^* : X --> X^d``.

    If the vector field space ``X^d`` is weighted by a vector ``v``,
    the adjoint, applied to a function ``h`` from ``X`` is the vector
    field

        ``x --> h(x) * (w / v) * G(x)``,

    where ``G`` and ``w`` are the vector field and weighting from the
    inner product operator, resp., and all multiplications are understood
    component-wise.
    """

    def __init__(self, sspace, vecfield, vfspace=None, weight=None):
        """Initialize a new instance.

        Parameters
        ----------
        sspace : `LinearSpace`
            "Scalar" space on which the operator acts
        vecfield : `range` `element-like`
            Vector field of the point-wise inner product operator
        vfspace : `ProductSpace`, optional
            Space of vector fields to which the operator maps. It must
            be a power space with ``sspace`` as base space.
            This option is intended to enforce an operator range
            with a certain weighting.
            Default: ``ProductSpace(space, len(vecfield), weight=weight)``
        weight : `array-like` or float, optional
            Weighting array or constant of the inner product operator.
            If an array is given, its length must be equal to
            ``len(vecfield)``, and all entries must be positive. A
            provided constant must be positive.
            By default, the weights are is taken from
            ``range.weighting`` if applicable. Note that this excludes
            unusual weightings with custom inner product, norm or dist.
        """
        if vfspace is None:
            vfspace = ProductSpace(sspace, len(vecfield), weight=weight)
        else:
            if not isinstance(vfspace, ProductSpace):
                raise TypeError('`vfspace` {!r} is not a '
                                'ProductSpace instance'.format(vfspace))
            if vfspace[0] != sspace:
                raise ValueError('base space of the range is different from '
                                 'the given scalar space ({!r} != {!r})'
                                 ''.format(vfspace[0], sspace))
        super().__init__(adjoint=True, vfspace=vfspace, vecfield=vecfield,
                         weight=weight)

        # Get weighting from range
        if hasattr(self.range.weighting, 'vector'):
            self._ran_weights = self.range.weighting.vector
        elif hasattr(self.range.weighting, 'const'):
            self._ran_weights = (self.range.weighting.const *
                                 np.ones(len(self.range)))
        else:
            raise ValueError('weighting scheme {!r} of the range does '
                             'not define a weighting vector or constant'
                             ''.format(self.range.weighting))

    def _call(self, f, out):
        """Implement ``self(vf, out)``."""
        for vfi, oi, ran_wi, dom_wi in zip(self.vecfield, out,
                                           self._ran_weights, self.weights):
            vfi.multiply(f, out=oi)
            if not np.isclose(ran_wi, dom_wi):
                oi *= dom_wi / ran_wi

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `PointwiseInner`
        """
        return PointwiseInner(vfspace=self.range, vecfield=self.vecfield,
                              weight=self.weights)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
