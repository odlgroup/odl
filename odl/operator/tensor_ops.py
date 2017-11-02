﻿# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Operators defined for tensor fields."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

import numpy as np

from odl.operator import Operator
from odl.set import RealNumbers, ComplexNumbers
from odl.space import ProductSpace, fn
from odl.space.base_ntuples import FnBase
from odl.util import writable_array, signature_string, indent


__all__ = ('PointwiseNorm', 'PointwiseInner', 'PointwiseSum', 'MatrixOperator')

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

    .. note::
        It is allowed that ``domain``, ``range`` and ``base_space`` use
        different ``dtype``. Correctness for, e.g., real-to-complex mappings
        is not guaranteed in that case.

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
            ``X`` (up to ``dtype``), or the base space itself.
            Empty product spaces are not allowed.
        base_space : `LinearSpace`
            The base space ``X``.
        linear : bool, optional
            If ``True``, assume that the operator is linear.
        """
        if not is_compatible_space(domain, base_space):
            raise ValueError(
                '`domain` {!r} is not compatible with `base_space` {!r}'
                ''.format(domain, base_space))

        if not is_compatible_space(range, base_space):
            raise ValueError(
                '`range` {!r} is not compatible with `base_space` {!r}'
                ''.format(range, base_space))

        super(PointwiseTensorFieldOperator, self).__init__(
            domain=domain, range=range, linear=linear)
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
        >>> pw_norm = odl.PointwiseNorm(vfspace)
        >>> pw_norm.range == spc
        True

        Now we can calculate the 2-norm in each point:

        >>> x = vfspace.element([[[1, -4]],
        ...                      [[0, 3]]])
        >>> print(pw_norm(x))
        [[ 1.,  5.]]

        We can change the exponent either in the vector field space
        or in the operator directly:

        >>> vfspace = odl.ProductSpace(spc, 2, exponent=1)
        >>> pw_norm = PointwiseNorm(vfspace)
        >>> print(pw_norm(x))
        [[ 1.,  7.]]
        >>> vfspace = odl.ProductSpace(spc, 2)
        >>> pw_norm = PointwiseNorm(vfspace, exponent=1)
        >>> print(pw_norm(x))
        [[ 1.,  7.]]
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`vfspace` {!r} is not a ProductSpace '
                            'instance'.format(vfspace))
        super(PointwiseNorm, self).__init__(
            domain=vfspace, range=vfspace[0], base_space=vfspace[0],
            linear=False)

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
            super(PointwiseInnerBase, self).__init__(
                domain=vfspace[0], range=vfspace, base_space=vfspace[0],
                linear=True)
        else:
            super(PointwiseInnerBase, self).__init__(
                domain=vfspace, range=vfspace[0], base_space=vfspace[0],
                linear=True)

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
        [[ 0., -7.]]
        """
        super(PointwiseInner, self).__init__(
            adjoint=False, vfspace=vfspace, vecfield=vecfield,
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
        super(PointwiseInnerAdjoint, self).__init__(
            adjoint=True, vfspace=vfspace, vecfield=vecfield,
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


# TODO: Make this an optimized operator on its own.
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
        [[ 1., -1.]]
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError('`vfspace` {!r} is not a ProductSpace '
                            'instance'.format(vfspace))

        ones = vfspace.one()
        super(PointwiseSum, self).__init__(
            vfspace, vecfield=ones, weighting=weighting)


class MatrixOperator(Operator):

    """Matrix-vector multiplication as a linear operator.

    This operator uses a matrix to represent an operator, and get its
    adjoint and inverse by doing computations on the matrix. This is in
    general a rather slow and memory-inefficient approach, and users are
    recommended to use other alternatives if possible.
    """

    def __init__(self, matrix, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        matrix : `array-like` or `scipy.sparse.base.spmatrix`
            2-dimensional array representing the linear operator.
        domain : `FnBase`, optional
            Space of elements on which the operator can act. Its
            ``dtype`` must be castable to ``range.dtype``.
            For the default ``None``, a `NumpyFn` space with size
            ``matrix.shape[1]`` is used, together with the matrix'
            data type.
        range : `FnBase`, optional
            Space of elements on to which the operator maps. Its
            ``shape`` and ``dtype`` attributes must match the ones
            of the result of the multiplication.
            For the default ``None``, the range is inferred from
            ``matrix`` and ``domain``.

        Examples
        --------
        By default, ``domain`` and ``range`` are `NumpyFn` type spaces:

        >>> matrix = np.ones((3, 4))
        >>> op = MatrixOperator(matrix)
        >>> op
        MatrixOperator(
            [[ 1.,  1.,  1.,  1.],
             [ 1.,  1.,  1.,  1.],
             [ 1.,  1.,  1.,  1.]]
        )
        >>> op.domain
        rn(4)
        >>> op.range
        rn(3)
        >>> op([1, 2, 3, 4])
        rn(3).element([ 10.,  10.,  10.])

        They can also be provided explicitly, for example with
        `uniform_discr` spaces:

        >>> dom = odl.uniform_discr(0, 1, 4)
        >>> ran = odl.uniform_discr(0, 1, 3)
        >>> op = MatrixOperator(matrix, domain=dom, range=ran)
        >>> op(dom.one())
        uniform_discr(0.0, 1.0, 3).element([ 4., 4., 4.])

        For storage efficiency, SciPy sparse matrices can be used:

        >>> import scipy
        >>> row_idcs = np.array([0, 3, 1, 0])
        >>> col_idcs = np.array([0, 3, 1, 2])
        >>> values = np.array([4.0, 5.0, 7.0, 9.0])
        >>> matrix = scipy.sparse.coo_matrix((values, (row_idcs, col_idcs)),
        ...                                  shape=(4, 4))
        >>> matrix.toarray()
        array([[ 4.,  0.,  9.,  0.],
               [ 0.,  7.,  0.,  0.],
               [ 0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  5.]])
        >>> op = MatrixOperator(matrix)
        >>> op(op.domain.one())
        rn(4).element([ 13.,   7.,   0.,   5.])
        """
        # Lazy import to improve `import odl` time
        import scipy.sparse

        if scipy.sparse.isspmatrix(matrix):
            self.__matrix = matrix
        else:
            self.__matrix = np.asarray(matrix)

        if self.matrix.ndim != 2:
            raise ValueError('matrix {} has {} axes instead of 2'
                             ''.format(matrix, self.matrix.ndim))

        # Infer domain and range from matrix if necessary
        if domain is None:
            domain = fn(self.matrix.shape[1], dtype=self.matrix.dtype)
        elif not isinstance(domain, FnBase):
            raise TypeError('`domain` {!r} is not an `FnBase` instance'
                            ''.format(domain))

        if range is None:
            range = fn(self.matrix.shape[0], dtype=self.matrix.dtype)
        elif not isinstance(range, FnBase):
            raise TypeError('`range` {!r} is not an `FnBase` instance'
                            ''.format(range))

        # Check compatibility of matrix with domain and range
        if self.matrix.shape != (range.size, domain.size):
            raise ValueError('matrix shape {} does not match the required '
                             'shape {} of a matrix {} --> {}'
                             ''.format(self.matrix.shape,
                                       (range.size, domain.size),
                                       domain, range))

        if not np.can_cast(domain.dtype, range.dtype):
            raise TypeError('domain data type {!r} cannot be safely cast to '
                            'range data type {!r}'
                            ''.format(domain.dtype, range.dtype))

        if not np.can_cast(self.matrix.dtype, range.dtype):
            raise TypeError('matrix data type {!r} cannot be safely cast to '
                            'range data type {!r}.'
                            ''.format(matrix.dtype, range.dtype))

        super(MatrixOperator, self).__init__(domain, range, linear=True)

    @property
    def matrix(self):
        """Matrix representing this operator."""
        return self.__matrix

    @property
    def matrix_issparse(self):
        """Whether the representing matrix is sparse or not."""
        # Lazy import to improve `import odl` time
        import scipy.sparse
        return scipy.sparse.isspmatrix(self.matrix)

    @property
    def adjoint(self):
        """Adjoint operator represented by the adjoint matrix.

        Returns
        -------
        adjoint : `MatrixOperator`
        """
        if self.domain.field != self.range.field:
            raise NotImplementedError('adjoint not defined since fields '
                                      'of domain and range differ ({} != {})'
                                      ''.format(self.domain.field,
                                                self.range.field))
        return MatrixOperator(self.matrix.conj().T,
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
        inverse : `MatrixOperator`
        """
        # Lazy import to improve `import odl` time
        import scipy.sparse

        if scipy.sparse.issparse(self.matrix):
            dense_matrix = self.matrix.toarray()
        else:
            dense_matrix = self.matrix
        return MatrixOperator(np.linalg.inv(dense_matrix),
                              domain=self.range, range=self.domain)

    def _call(self, x, out=None):
        """Return ``self(x[, out])``."""
        if out is None:
            return self.range.element(self.matrix.dot(x))
        else:
            if self.matrix_issparse:
                # Unfortunately, there is no native in-place dot product for
                # sparse matrices
                out[:] = self.matrix.dot(x)
            else:
                with writable_array(out) as out_arr:
                    self.matrix.dot(x, out=out_arr)

    def __repr__(self):
        """Return ``repr(self)``."""
        # Matrix printing itself in an executable way (for dense matrix)
        if self.matrix_issparse:
            # Don't convert to dense, can take forever
            matrix_str = repr(self.matrix)
        else:
            matrix_str = np.array2string(self.matrix, separator=', ')
        posargs = [matrix_str]

        # Optional arguments with defaults, inferred from the matrix
        optargs = []
        # domain
        optargs.append(
            ('domain', self.domain, fn(self.matrix.shape[1],
                                       self.matrix.dtype))
        )
        # range
        optargs.append(
            ('range', self.range, fn(self.matrix.shape[0],
                                     self.matrix.dtype))
        )

        inner_str = signature_string(posargs, optargs, sep=[', ', ', ', ',\n'],
                                     mod=[['!s'], ['!r', '!r']])
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


def is_compatible_space(space, base_space):
    """Check compatibility of a (power) space with a base space.

    Compatibility here means that the spaces are equal or ``space``
    is a non-empty power space of ``base_space`` up to different
    data types.

    Parameters
    ----------
    space, base_space : `LinearSpace`
        Spaces to check for compatibility. ``base_space`` cannot be a
        `ProductSpace`.

    Returns
    -------
    is_compatible : bool
        ``True`` if

        - ``space == base_space`` or
        - ``space.astype(base_space.dtype) == base_space``, provided that
          these properties exist, or
        - ``space`` is a power space of nonzero length and one of the three
          situations applies to ``space[0]`` (recursively).

        Otherwise ``False``.

    Examples
    --------
    Scalar spaces:

    >>> base = odl.rn(2)
    >>> is_compatible_space(odl.rn(2), base)
    True
    >>> is_compatible_space(odl.rn(3), base)
    False
    >>> is_compatible_space(odl.rn(2, dtype='float32'), base)
    True

    Power spaces:

    >>> is_compatible_space(odl.rn(2) ** 2, base)
    True
    >>> is_compatible_space(odl.rn(2) * odl.rn(3), base)  # no power space
    False
    >>> is_compatible_space(odl.rn(2, dtype='float32') ** 2, base)
    True
    """
    if isinstance(base_space, ProductSpace):
        return False

    if isinstance(space, ProductSpace):
        if not space.is_power_space:
            return False
        elif len(space) == 0:
            return False
        else:
            return is_compatible_space(space[0], base_space)
    else:
        if hasattr(space, 'astype') and hasattr(base_space, 'dtype'):
            # TODO: maybe only the shape should play a role?
            comp_space = space.astype(base_space.dtype)
        else:
            comp_space = space

        return comp_space == base_space


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
