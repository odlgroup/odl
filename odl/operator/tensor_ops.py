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


__all__ = ('PointwiseNorm', 'PointwiseInner', 'PointwiseSum', 'MatVecOperator')

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
    ``d``. It is also possible to have tensor fields over tensor fields, i.e.
    ``ProductSpace(ProductSpace(X, n), m)``.

    See Also
    --------
    ProductSpace
    """

    def __init__(self, domain, range, base_space, linear=False):
        """Initialize a new instance.

        Parameters
        ----------
        domain, range : {`ProductSpace`, `LinearSpace`}
            Spaces of vector fields between which the operator maps.
            They have to be either power spaces of the same base space
            ``X`` or the base space itself (only one of them).
            Empty product spaces are not allowed.
        base_space : `LinearSpace`
            The base space ``X``.
        linear : bool, optional
            If ``True``, assume that the operator is linear.
        """
        if domain != base_space:
            if (not isinstance(domain, ProductSpace) or
                    not domain.is_power_space or
                    domain.size == 0):
                raise TypeError('`domain` {!r} is neither `base_space` {!r} '
                                'nor a nonempty power space of it'
                                ''.format(domain, base_space))

        if range != base_space:
            if (not isinstance(range, ProductSpace) or
                    not range.is_power_space or
                    range.size == 0 or
                    range[0] != base_space):
                raise TypeError('`range` {!r} is neither `base_space` {!r} '
                                'nor a nonempty power space of it'
                                ''.format(range, base_space))

        super().__init__(domain=domain, range=range, linear=linear)
        self.__base_space = base_space

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

    def __init__(self, vfspace, exponent=None, weighting=None):
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
        weighting : `array-like` or positive float, optional
            Weighting array or constant for the norm. If an array is
            given, its length must be equal to ``domain.size``, and
            all entries must be positive.
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
        super().__init__(domain=vfspace, range=vfspace[0],
                         base_space=vfspace[0], linear=False)

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
        if weighting is None:
            # TODO: find a more robust way of getting the weights as an array
            if hasattr(self.domain.weighting, 'array'):
                self.__weights = self.domain.weighting.array
            elif hasattr(self.domain.weighting, 'const'):
                self.__weights = (self.domain.weighting.const *
                                  np.ones(len(self.domain)))
            else:
                raise ValueError('weighting scheme {!r} of the domain does '
                                 'not define a weighting array or constant'
                                 ''.format(self.domain.weighting))
        elif np.isscalar(weighting):
            if weighting <= 0:
                raise ValueError('weighting constant must be positive, got '
                                 '{}'.format(weighting))
            self.__weights = float(weighting) * np.ones(self.domain.size)
        else:
            self.__weights = np.asarray(weighting, dtype='float64')
            if (not np.all(self.weights > 0) or
                    not np.all(np.isfinite(self.weights))):
                raise ValueError('weighting array {} contains invalid '
                                 'entries'.format(weighting))
        self.__is_weighted = not np.array_equiv(self.weights, 1.0)

    @property
    def exponent(self):
        """Exponent ``p`` of this norm."""
        return self._exponent

    @property
    def weights(self):
        """Weighting array of this operator."""
        return self.__weights

    @property
    def is_weighted(self):
        """``True`` if weighting is not 1 or all ones."""
        return self.__is_weighted

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
        vf[0].ufuncs.absolute(out=out)
        if self.is_weighted:
            out *= self.weights[0]

        if self.domain.size == 1:
            return

        tmp = self.range.element()
        for fi, wi in zip(vf[1:], self.weights[1:]):
            fi.ufuncs.absolute(out=tmp)
            if self.is_weighted:
                tmp *= wi
            out += tmp

    def _call_vecfield_inf(self, vf, out):
        """Implement ``self(vf, out)`` for exponent ``inf``."""
        vf[0].ufuncs.absolute(out=out)
        if self.is_weighted:
            out *= self.weights[0]

        if self.domain.size == 1:
            return

        tmp = self.range.element()
        for vfi, wi in zip(vf[1:], self.weights[1:]):
            vfi.ufuncs.absolute(out=tmp)
            if self.is_weighted:
                tmp *= wi
            out.ufuncs.maximum(tmp, out=out)

    def _call_vecfield_p(self, vf, out):
        """Implement ``self(vf, out)`` for exponent 1 < p < ``inf``."""
        # Optimization for 1 component - just absolute value (maybe weighted)
        if self.domain.size == 1:
            vf[0].ufuncs.absolute(out=out)
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

        out.ufuncs.power(1 / self.exponent, out=out)

    def _abs_pow_ufunc(self, fi, out):
        """Compute |F_i(x)|^p point-wise and write to ``out``."""
        # Optimization for a very common case
        if self.exponent == 2.0 and self.base_space.field == RealNumbers():
            fi.multiply(fi, out=out)
        else:
            fi.ufuncs.absolute(out=out)
            out.ufuncs.power(self.exponent, out=out)

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
        if self.exponent != 2:  # optimize away most common case.
            vf_pwnorm_fac **= (self.exponent - 1)

        inner_vf = vf.copy()

        for gi in inner_vf:
            gi /= vf_pwnorm_fac * gi ** (self.exponent - 2)

        return PointwiseInner(self.domain, inner_vf, weighting=self.weights)


class PointwiseInnerBase(PointwiseTensorFieldOperator):
    """Base class for `PointwiseInner` and `PointwiseInnerAdjoint`.

    Implemented to allow code reuse between the classes.
    """

    def __init__(self, adjoint, vfspace, vecfield, weighting=None):
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
        weighting : `array-like` or float, optional
            Weighting array or constant for the norm. If an array is
            given, its length must be equal to ``domain.size``.
            By default, the weights are is taken from
            ``domain.weighting``. Note that this excludes unusual
            weightings with custom inner product, norm or dist.
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`vfsoace` {!r} is not a ProductSpace '
                            'instance'.format(vfspace))
        if adjoint:
            super().__init__(domain=vfspace[0], range=vfspace,
                             base_space=vfspace[0], linear=True)
        else:
            super().__init__(domain=vfspace, range=vfspace[0],
                             base_space=vfspace[0], linear=True)

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
        if weighting is None:
            if hasattr(vfspace.weighting, 'array'):
                self.__weights = vfspace.weighting.array
            elif hasattr(vfspace.weighting, 'const'):
                self.__weights = (vfspace.weighting.const *
                                  np.ones(len(vfspace)))
            else:
                raise ValueError('weighting scheme {!r} of the domain does '
                                 'not define a weighting array or constant'
                                 ''.format(vfspace.weighting))
        elif np.isscalar(weighting):
            self.__weights = float(weighting) * np.ones(vfspace.size)
        else:
            self.__weights = np.asarray(weighting, dtype='float64')
        self.__is_weighted = not np.array_equiv(self.weights, 1.0)

    @property
    def vecfield(self):
        """Fixed vector field ``G`` of this inner product."""
        return self._vecfield

    @property
    def weights(self):
        """Weighting array of this operator."""
        return self.__weights

    @property
    def is_weighted(self):
        """``True`` if weighting is not 1 or all ones."""
        return self.__is_weighted

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

    def __init__(self, vfspace, vecfield, weighting=None):
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
        weighting : `array-like` or float, optional
            Weighting array or constant for the norm. If an array is
            given, its length must be equal to ``domain.size``, and
            all entries must be positive.
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
                         weighting=weighting)

    @property
    def vecfield(self):
        """Fixed vector field ``G`` of this inner product."""
        return self._vecfield

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
            vfspace=self.domain, weighting=self.weights)


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

    def __init__(self, sspace, vecfield, vfspace=None, weighting=None):
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
            Default: ``ProductSpace(space, len(vecfield),
            weighting=weighting)``
        weighting : `array-like` or float, optional
            Weighting array or constant of the inner product operator.
            If an array is given, its length must be equal to
            ``len(vecfield)``.
            By default, the weights are is taken from
            ``range.weighting`` if applicable. Note that this excludes
            unusual weightings with custom inner product, norm or dist.
        """
        if vfspace is None:
            vfspace = ProductSpace(sspace, len(vecfield), weighting=weighting)
        else:
            if not isinstance(vfspace, ProductSpace):
                raise TypeError('`vfspace` {!r} is not a '
                                'ProductSpace instance'.format(vfspace))
            if vfspace[0] != sspace:
                raise ValueError('base space of the range is different from '
                                 'the given scalar space ({!r} != {!r})'
                                 ''.format(vfspace[0], sspace))
        super().__init__(adjoint=True, vfspace=vfspace, vecfield=vecfield,
                         weighting=weighting)

        # Get weighting from range
        if hasattr(self.range.weighting, 'array'):
            self.__ran_weights = self.range.weighting.array
        elif hasattr(self.range.weighting, 'const'):
            self.__ran_weights = (self.range.weighting.const *
                                  np.ones(len(self.range)))
        else:
            raise ValueError('weighting scheme {!r} of the range does '
                             'not define a weighting array or constant'
                             ''.format(self.range.weighting))

    def _call(self, f, out):
        """Implement ``self(vf, out)``."""
        for vfi, oi, ran_wi, dom_wi in zip(self.vecfield, out,
                                           self.__ran_weights, self.weights):
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
                              weighting=self.weights)


# TODO: Optimize this to an optimized operator on its own.
class PointwiseSum(PointwiseInner):

    """Take the point-wise sum of a vector field.

    This operator takes the (weighted) sum

        ``sum(F(x)) = [ sum_j( w_j * F_j(x) ) ]

    where ``F`` is a vector field. This implies that
    the `Operator.domain` is a power space of a discretized function
    space. For example, if ``X`` is a `DiscreteLp` space, then
    ``ProductSpace(X, d)`` is a valid domain for any positive integer
    ``d``.
    """

    def __init__(self, vfspace, weighting=None):
        """Initialize a new instance.

        Parameters
        ----------
        vfspace : `ProductSpace`
            Space of vector fields on which the operator acts.
            It has to be a product space of identical spaces, i.e. a
            power space.
        weighting : `array-like` or float, optional
            Weighting array or constant for the sum. If an array is
            given, its length must be equal to ``domain.size``.
            By default, the weights are is taken from
            ``domain.weighting``. Note that this excludes unusual
            weightings with custom inner product, norm or dist.

        Examples
        --------
        We make a tiny vector field space in 2D and create the
        standard point-wise sum operator on that space. The operator
        maps a vector field to a scalar function:

        >>> spc = odl.uniform_discr([-1, -1], [1, 1], (1, 2))
        >>> vfspace = odl.ProductSpace(spc, 2)
        >>> pw_sum = PointwiseSum(vfspace)
        >>> pw_sum.range == spc
        True

        Now we can calculate the sum in each point:

        >>> x = vfspace.element([[[1, -4]],
        ...                      [[0, 3]]])
        >>> print(pw_sum(x))
        [[1.0, -1.0]]
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`vfspace` {!r} is not a ProductSpace '
                            'instance'.format(vfspace))

        ones = vfspace.one()
        super().__init__(vfspace, vecfield=ones, weighting=weighting)


class MatVecOperator(Operator):

    """Matrix multiply operator :math:`\mathbb{F}^n -> \mathbb{F}^m`.

    This operator uses a matrix to represent an operator, and get its adjoint
    and inverse by doing computations on the matrix. This is in general a
    rather slow approach, and users are recommended to use other alternatives
    if available.
    """

    def __init__(self, matrix, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        matrix : `array-like` or  `scipy.sparse.spmatrix`
            Matrix representing the linear operator. Its shape must be
            ``(m, n)``, where ``n`` is the size of ``domain`` and ``m`` the
            size of ``range``. Its dtype must be castable to the range
            ``dtype``.
        domain : `NumpyFn`, optional
            Space on whose elements the matrix acts. If not provided,
            the domain is inferred from the matrix ``dtype`` and
            ``shape``. If provided, its dtype must be castable to the
            range dtype.
        range : `NumpyFn`, optional
            Space to which the matrix maps. If not provided,
            the domain is inferred from the matrix ``dtype`` and
            ``shape``.
        """
        # TODO: fix dead link `scipy.sparse.spmatrix`
        if isspmatrix(matrix):
            self.__matrix = matrix
        else:
            self.__matrix = np.asarray(matrix)

        if self.matrix.ndim != 2:
            raise ValueError('matrix {} has {} axes instead of 2'
                             ''.format(matrix, self.matrix.ndim))

        # Infer domain and range from matrix if necessary
        if domain is None:
            domain = NumpyFn(self.matrix.shape[1], dtype=self.matrix.dtype)
        elif not isinstance(domain, NumpyFn):
            raise TypeError('`domain` {!r} is not an `NumpyFn` instance'
                            ''.format(domain))

        if range is None:
            range = NumpyFn(self.matrix.shape[0], dtype=self.matrix.dtype)
        elif not isinstance(range, NumpyFn):
            raise TypeError('`range` {!r} is not an `NumpyFn` instance'
                            ''.format(range))

        # Check compatibility of matrix with domain and range
        if not np.can_cast(domain.dtype, range.dtype):
            raise TypeError('domain data type {!r} cannot be safely cast to '
                            'range data type {!r}'
                            ''.format(domain.dtype, range.dtype))

        if self.matrix.shape != (range.size, domain.size):
            raise ValueError('matrix shape {} does not match the required '
                             'shape {} of a matrix {} --> {}'
                             ''.format(self.matrix.shape,
                                       (range.size, domain.size),
                                       domain, range))
        if not np.can_cast(self.matrix.dtype, range.dtype):
            raise TypeError('matrix data type {!r} cannot be safely cast to '
                            'range data type {!r}.'
                            ''.format(matrix.dtype, range.dtype))

        super().__init__(domain, range, linear=True)

    @property
    def matrix(self):
        """Matrix representing this operator."""
        return self.__matrix

    @property
    def matrix_issparse(self):
        """Whether the representing matrix is sparse or not."""
        return isspmatrix(self.matrix)

    @property
    def adjoint(self):
        """Adjoint operator represented by the adjoint matrix.

        Returns
        -------
        adjoint : `MatVecOperator`
        """
        if self.domain.field != self.range.field:
            raise NotImplementedError('adjoint not defined since fields '
                                      'of domain and range differ ({} != {})'
                                      ''.format(self.domain.field,
                                                self.range.field))
        return MatVecOperator(self.matrix.conj().T,
                              domain=self.range, range=self.domain)

    @property
    def inverse(self):
        """Inverse operator represented by the inverse matrix.

        Taking the inverse causes sparse matrices to become dense and is
        generally very heavy computationally since the matrix is inverted
        numerically (an O(n^3) operation). It is recommended to instead
        use one of the solvers available in the ``odl.solvers`` package.

        Returns
        -------
        inverse : `MatVecOperator`
        """
        if self.matrix_issparse:
            dense_matrix = self.matrix.toarray()
        else:
            dense_matrix = self.matrix

        return MatVecOperator(np.linalg.inv(dense_matrix),
                              domain=self.range, range=self.domain)

    def _call(self, x, out=None):
        """Raw apply method on input, writing to given output."""
        if out is None:
            return self.range.element(self.matrix.dot(x.data))
        else:
            if self.matrix_issparse:
                # Unfortunately, there is no native in-place dot product for
                # sparse matrices
                out.data[:] = self.matrix.dot(x.data)
            else:
                self.matrix.dot(x.data, out=out.data)

    # TODO: repr and str



if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
