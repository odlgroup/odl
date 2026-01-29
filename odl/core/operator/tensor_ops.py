# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Operators defined for tensor fields."""


from numbers import Integral
from typing import Optional, Iterable
from dataclasses import dataclass

import numpy as np

from odl.core.util.npy_compat import AVOID_UNNECESSARY_COPY

from odl.core.operator.operator import Operator, AdapterOperator
from odl.core.operator.default_ops import IdentityOperator
from odl.core.operator.pspace_ops import DiagonalOperator
from odl.core.set import ComplexNumbers, RealNumbers
from odl.core.set.space import LinearSpace
from odl.core.space import ProductSpace, tensor_space
from odl.core.space.base_tensors import TensorSpace, Tensor
from odl.core.space.weightings.weighting import ArrayWeighting
from odl.core.util import dtype_repr, indent, signature_string
from odl.core.array_API_support import (ArrayBackend, lookup_array_backend,
                                        abs as odl_abs, maximum, pow, sqrt,
                                        multiply, get_array_and_backend,
                                        can_cast, odl_all_equal)

from odl.core.sparse import is_sparse, get_sparse_matrix_impl, lookup_sparse_format

__all__ = ('PointwiseNorm', 'PointwiseInner', 'PointwiseSum', 'MatrixOperator',
           'SamplingOperator', 'WeightedSumSamplingOperator',
           'FlatteningOperator', 'DeviceChange', 'ArrayBackendChange')

_SUPPORTED_DIFF_METHODS = ('central', 'forward', 'backward')

class _ImplChangeOperator(Operator):
    """An operator that is mathematically the identity, but whose domain and codomain
    differ in what backend or device they use for their arrays.
    This class is not intended for direct use (it is tedious having to explicitly
    define the domain and range as identical spaces with different implementation).
    Instead, use `ArrayBackendChange` and `DeviceChange`, which only require specifying
    the actual change that needs to happen. Both are automatically converted to
    `_ImplChangeOperator` when used in an operator pipeline.
    """

    def __init__(self, domain, range):
        """Create an operator tying two equivalent spaces with different storage together.

        Parameters
        ----------
        domain, range : `TensorSpace`
            Spaces of vectors. They must be identical save for the backend (`impl`)
            or the device.
        """
        assert(domain.shape == range.shape)
        assert(domain.impl == range.impl or domain.device == range.device)
        super().__init__(domain, range=range, linear=True)

    def _call(self, x):
        """Copy data to the intended backend."""
        if self.range.impl != self.domain.impl:
            return x.to_impl(self.range.impl)
        elif self.range.device != self.domain.device:
            return x.to_device(self.range.device)
        else:
            return x

    @property
    def inverse(self):
        """Operator that copies data back to the original backend."""
        return _ImplChangeOperator(domain=self.range, range=self.domain)

    @property
    def adjoint(self):
        """Adjoint is the same as inverse, as backend change is mathematically
        the identity."""
        return self.inverse

    def norm(self, estimate=False, **kwargs):
        """Return the operator norm of this operator. This is 1, as the
        operator is mathematically the identity."""
        return 1

    def __repr__(self):
        """Represent the operator by its domain and the impl of the range."""
        return f"{self.__class__.__name__}(domain={repr(self.domain)}, range_impl={repr(self.range.impl)})"

    def __str__(self):
        return f"{self.__class__.__name__}(domain={str(self.domain)}, range_impl={str(self.range.impl)})"

@dataclass(repr=True)
class ProductSpaceOverindexingException(ValueError):
    space: LinearSpace
    subspace_index: int | list[int]
    def __str__(self):
        return repr(self)

class DeviceChange(AdapterOperator):
    """A pseudo-operator that copies arrays from one computational device to another.
    This is useful as an adapter in a pipeline of operators that need to use different
    devices for some reason.
    Note that it is usually more efficient to implement your whole pipeline on a single
    device, if possible.
    """

    def __init__(self, domain_device: str, range_device: str,
                 subspace_index: int | list[int] =[]):
        """Create an operator tying two equivalent spaces with different storage together.

        Parameters
        ----------
        domain_device, range_device : `str`
            Device specifiers such as `'cpu'` or `'cuda:0'`. Which ones are
            supported depends on the backend and hardware.
        subspace_index: int or sequence of ints
            If the domain is a compound space (i.e. `ProductSpace`), you may wish
            to only change the device of one of the constituent spaces. This can
            be done with this index into the product space structure.
            If a list is provided, this is understood as recursive indexing into
            a nested product space.
        """
        self._domain_device = domain_device
        self._range_device = range_device

        # TODO refactor logic for `subspace_index` to avoid duplication between
        # `DeviceChange` and `ArrayBackendChange`
        if isinstance(subspace_index, int):
            self._subspace_index = [subspace_index]
        elif (isinstance(subspace_index, Iterable)
                and all(isinstance(i,int) for i in subspace_index)):
            self._subspace_index = list(subspace_index)
        else:
            raise TypeError(
               f"`subspace_index` must be `int` or list of ints; got {type(subspace_index)}")

    def _subspace_index_exception(self, space):
        return ProductSpaceOverindexingException(
                   space=space, subspace_index=self._subspace_index)

    def _infer_op_from_domain(self, domain: LinearSpace) -> Operator:
        if isinstance(domain, ProductSpace):
            if self._subspace_index:
                subchanger = DeviceChange(domain_device=self._domain_device,
                                          range_device=self._range_device,
                                          subspace_index=self._subspace_index[1:])
                try:
                    return DiagonalOperator(*[subchanger._infer_op_from_domain(p)
                                               if i==self._subspace_index[0]
                                               else IdentityOperator(p)
                                              for i,p in enumerate(domain.spaces)])
                except ProductSpaceOverindexingException as e:
                    raise self._subspace_index_exception(domain) from e
            else:
                return DiagonalOperator(*[self._infer_op_from_domain(p) for p in domain.spaces])
        elif not isinstance(domain, TensorSpace):
            raise TypeError(f"Device change is only defined on `TensorSpace` or `ProductSpace`.")
        elif domain.device != self._domain_device:
            raise ValueError(f"Expected {self._domain_device}, got {domain.device=}")
        elif len(self._subspace_index) > 0:
            raise self._subspace_index_exception(domain)
        else:
            return _ImplChangeOperator(domain=domain, range=domain.to_device(self._range_device))

    def _infer_op_from_range(self, range: LinearSpace) -> Operator:
        index_exception = ProductSpaceOverindexingException(
                            space=range, subspace_index=self._subspace_index)
        if isinstance(range, ProductSpace):
            if self._subspace_index:
                subchanger = DeviceChange(domain_device=self._domain_device,
                                          range_device=self._range_device,
                                          subspace_index=self._subspace_index[1:])
                try:
                    return DiagonalOperator(*[subchanger._infer_op_from_range(p)
                                               if i==self._subspace_index[0]
                                               else IdentityOperator(p)
                                              for i,p in enumerate(range.spaces)])
                except ProductSpaceOverindexingException as e:
                    raise self._subspace_index_exception(range) from e
            else:
                return DiagonalOperator(*[self._infer_op_from_range(p) for p in range.spaces])
        elif not isinstance(range, TensorSpace):
            raise TypeError(f"Device change is only defined on `TensorSpace` or `ProductSpace`.")
        elif range.device != self._range_device:
            raise ValueError(f"Expected {self._range_device}, got {range.device=}")
        elif len(self._subspace_index) > 0:
            raise self._subspace_index_exception(range)
        else:
            return _ImplChangeOperator(domain=range.to_device(self._domain_device), range=range)

    def __repr__(self):
        """Represent the operator by its domain and the device of the range."""
        return f"{self.__class__.__name__}(domain_device={repr(self._domain_device)}, range_device={repr(self._range_device)})"

    def __str__(self):
        return f"{self.__class__.__name__}(domain_device={str(self._domain_device)}, range_device={str(self._range_device)})"

class ArrayBackendChange(AdapterOperator):
    """A pseudo-operator that transfers arrays from one backend to another.
    Both backends must support the same device (this can mean you first need to use
    `DeviceChange` to transfer to `'cpu'`, which should be supported by all backends).
    """

    def __init__(self, domain_impl: str, range_impl: str,
                 subspace_index: int | list[int] =[]):
        """Create an operator tying two equivalent spaces with different storage together.

        Parameters
        ----------
        domain_impl, range_impl : `str`
            Backend specifiers such as `'numpy'` or `'pytorch'`. Which ones are
            supported depends on the installed packages.
        """
        self._domain_impl = domain_impl
        self._range_impl = range_impl

        if isinstance(subspace_index, int):
            self._subspace_index = [subspace_index]
        elif (isinstance(subspace_index, Iterable)
                and all(isinstance(i,int) for i in subspace_index)):
            self._subspace_index = list(subspace_index)
        else:
            raise TypeError(
               f"`subspace_index` must be `int` or list of ints; got {type(subspace_index)}")

    def _subspace_index_exception(self, space):
        return ProductSpaceOverindexingException(
                   space=space, subspace_index=self._subspace_index)

    def _infer_op_from_domain(self, domain: LinearSpace) -> Operator:
        if isinstance(domain, ProductSpace):
            if self._subspace_index:
                subchanger = ArrayBackendChange(domain_impl=self._domain_impl,
                                          range_impl=self._range_impl,
                                          subspace_index=self._subspace_index[1:])
                try:
                    return DiagonalOperator(*[subchanger._infer_op_from_domain(p)
                                               if i==self._subspace_index[0]
                                               else IdentityOperator(p)
                                              for i,p in enumerate(domain.spaces)])
                except ProductSpaceOverindexingException as e:
                    raise self._subspace_index_exception(range) from e
            else:
                return DiagonalOperator(*[self._infer_op_from_domain(p) for p in domain.spaces])
        elif not isinstance(domain, TensorSpace):
            raise TypeError(f"Backend change is only defined on `TensorSpace` or `ProductSpace`.")
        elif domain.impl != self._domain_impl:
            raise ValueError(f"Expected {self._domain_impl}, got {domain.impl=}")
        elif len(self._subspace_index) > 0:
            raise self._subspace_index_exception(domain)
        else:
            return _ImplChangeOperator(domain=domain, range=domain.to_impl(self._range_impl))

    def _infer_op_from_range(self, range: LinearSpace) -> Operator:
        if isinstance(range, ProductSpace):
            if self._subspace_index:
                subchanger = ArrayBackendChange(domain_impl=self._domain_impl,
                                          range_impl=self._range_impl,
                                          subspace_index=self._subspace_index[1:])
                try:
                    return DiagonalOperator(*[subchanger._infer_op_from_range(p)
                                               if i==self._subspace_index[0]
                                               else IdentityOperator(p)
                                              for i,p in enumerate(range.spaces)])
                except ProductSpaceOverindexingException as e:
                    raise self._subspace_index_exception(range) from e
            else:
                return DiagonalOperator(*[self._infer_op_from_range(p) for p in range.spaces])
        elif not isinstance(range, TensorSpace):
            raise TypeError(f"Backend change is only defined on `TensorSpace` or `ProductSpace`.")
        elif range.impl != self._range_impl:
            raise ValueError(f"Expected {self._range_impl}, got {range.impl=}")
        elif len(self._subspace_index) > 0:
            raise self._subspace_index_exception(range)
        return _ImplChangeOperator(domain=range.to_impl(self._domain_impl), range=range)

    def norm(self, estimate=False, **kwargs):
        """Return the operator norm of this operator. This is 1, as the
        operator is mathematically the identity."""
        return 1

    def __repr__(self):
        """Represent the operator by its domain and the device of the range."""
        return f"{self.__class__.__name__}(domain_impl={repr(self._domain_impl)}, range_impl={repr(self._range_impl)})"

    def __str__(self):
        return f"{self.__class__.__name__}(domain_impl={str(self._domain_impl)}, range_impl={str(self._range_impl)})"



class PointwiseTensorFieldOperator(Operator):
    """Abstract operator for point-wise tensor field manipulations.

    A point-wise operator acts on a space of vector or tensor fields,
    i.e. a power space ``X^d`` of a discretized function space ``X``.
    Its range is the power space ``X^k`` with a possibly different
    number ``k`` of components. For ``k == 1``, the base space
    ``X`` can be used instead.

    For example, if ``X`` is a `DiscretizedSpace` space, then
    ``ProductSpace(X, d)`` is a valid domain for any positive integer
    ``d``. It is also possible to have tensor fields over tensor fields, i.e.
    ``ProductSpace(ProductSpace(X, n), m)``.

    .. note::
        It is allowed that ``domain``, ``range`` and ``base_space`` use
        different ``dtype``. Correctness for, e.g., real-to-complex mappings
        is not guaranteed in that case.

    See Also
    --------
    odl.core.space.pspace.ProductSpace
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
                f"`domain` {domain} is not compatible with `base_space` {base_space}"
            )

        if not is_compatible_space(range, base_space):
            raise ValueError(
                f"`range` {range} is not compatible with `base_space` {base_space}"
            )

        super().__init__(domain=domain, range=range, linear=linear)
        self.__base_space = base_space

    @property
    def base_space(self):
        """Base space ``X`` of this operator's domain and range."""
        return self.__base_space


class PointwiseNorm(PointwiseTensorFieldOperator):
    """Take the point-wise norm of a vector field.

    This operator computes the (weighted) p-norm in each point of
    a vector field, thus producing a scalar-valued function.
    It implements the formulas ::

        ||F(x)|| = [ sum_j( w_j * |F_j(x)|^p ) ]^(1/p)

    for ``p`` finite and ::

        ||F(x)|| = max_j( w_j * |F_j(x)| )

    for ``p = inf``, where ``F`` is a vector field. This implies that
    the `Operator.domain` is a power space of a discretized function
    space. For example, if ``X`` is a `DiscretizedSpace` space, then
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
            given, its length must be equal to ``len(domain)``, and
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
            raise TypeError(f"`vfspace` {vfspace} is not a ProductSpace instance")
        super().__init__(
            domain=vfspace,
            range=vfspace[0].real_space,
            base_space=vfspace[0],
            linear=False,
        )

        # Need to check for product space shape once higher order tensors
        # are implemented

        if exponent is None:
            if self.domain.exponent is None:
                raise ValueError(f"cannot determine `exponent` from {self.domain}")
            self._exponent = self.domain.exponent
        elif exponent < 1:
            raise ValueError("`exponent` smaller than 1 not allowed")
        else:
            self._exponent = float(exponent)

        # Handle weighting, including sanity checks
        if weighting is None:
            # TODO: find a more robust way of getting the weights as an array
            if hasattr(self.domain.weighting, 'array'):
                self.__weights = self.domain.weighting.array
            elif hasattr(self.domain.weighting, 'const'):
                self.__weights = [
                    self.domain.weighting.const * self.domain[i].one()
                    for i in range(len(vfspace)) ]
            else:
                raise ValueError(
                    f"weighting scheme {self.domain.weighting} of the domain does not define a weighting array or constant"
                )
            self.__is_weighted = False

        else:
            ### This is a bad situation: although we worked hard to get an elegant weighting, the PointwiseNorm just yanks all of that down the drain by reimplementing the norm operation and the input sanitisation just for a ProductSpace.
            ### EV reimplemented these two functionnalities but moving forward, this should be coerced into abiding to our new API

            if (isinstance(weighting, list)
                 and all([isinstance(w, Tensor) for w in weighting])):
                self.__weights = weighting
                self.__is_weighted = all(odl_all_equal(w, 1) for w in weighting)
            else:
                if isinstance(weighting, (int, float)):
                    weighting = [weighting for _ in range(len(self.domain))]

                weighted_flag = []
                for i in range(len(self.domain)):
                    if weighting[i] <= 0:
                        raise ValueError(
                            f"weighting array weighting contains invalid entry {weighting[i]}"
                        )
                    if weighting[i] in [1, 1.0]:
                        weighted_flag.append(False)
                    else:
                        weighted_flag.append(True)
                self.__is_weighted = bool(any(weighted_flag))

                weighting = [
                    self.domain[i].tspace.broadcast_to(weighting[i])
                    for i in range(len(self.domain))
                ]

                self.__weights = []
                for i in range(len(self.domain)):
                    self.__weights.append(self.domain[i].element(weighting[i]))

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

        odl_abs(vf[0], out=out)
        if self.is_weighted:
            out *= self.weights[0]

        if len(self.domain) == 1:
            return

        tmp = self.range.element()
        for fi, wi in zip(vf[1:], self.weights[1:]):
            odl_abs(fi, out=tmp)
            if self.is_weighted:
                tmp *= wi
            out += tmp

    def _call_vecfield_inf(self, vf, out):
        """Implement ``self(vf, out)`` for exponent ``inf``."""
        odl_abs(vf[0], out=out)
        if self.is_weighted:
            out *= self.weights[0]

        if len(self.domain) == 1:
            return

        tmp = self.range.element()
        for vfi, wi in zip(vf[1:], self.weights[1:]):
            odl_abs(vfi, out=tmp)
            if self.is_weighted:
                tmp *= wi
            maximum(out, tmp, out=out)

    def _call_vecfield_p(self, vf, out):
        """Implement ``self(vf, out)`` for exponent 1 < p < ``inf``."""
        # Optimization for 1 component - just absolute value (maybe weighted)
        if len(self.domain) == 1:
            odl_abs(vf[0], out=out)
            if self.is_weighted:
                out *= self.weights[0] ** (1 / self.exponent)
            return

        # Initialize out, avoiding one copy
        self._abs_pow(vf[0], out=out, p=self.exponent)
        if self.is_weighted:
            out *= self.weights[0]

        tmp = self.range.element()
        for fi, wi in zip(vf[1:], self.weights[1:]):
            self._abs_pow(fi, out=tmp, p=self.exponent)
            if self.is_weighted:
                tmp *= wi
            out += tmp

        self._abs_pow(out, out=out, p=(1 / self.exponent))

    def _abs_pow(self, fi, out, p):
        """Compute |F_i(x)|^p point-wise and write to ``out``."""
        # Optimization for very common cases
        if p == 0.5:
            odl_abs(fi, out=out)
            sqrt(out, out=out)
        elif p == 2.0 and self.base_space.field == RealNumbers():
            multiply(fi, fi, out=out)
        else:
            odl_abs(fi, out=out)
            pow(out, p, out=out)

    def derivative(self, vf):
        """Derivative of the point-wise norm operator at ``vf``.

        The derivative at ``F`` of the point-wise norm operator ``N``
        with finite exponent ``p`` and weights ``w`` is the pointwise
        inner product with the vector field ::

            x --> N(F)(x)^(1-p) * [ F_j(x) * |F_j(x)|^(p-2) ]_j

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
            raise NotImplementedError("operator not Frechet-differentiable "
                                      "on a complex space")

        if self.exponent == float('inf'):
            raise NotImplementedError("operator not Frechet-differentiable "
                                      "for exponent = inf")

        vf = self.domain.element(vf)
        vf_pwnorm_fac = self(vf)
        if self.exponent != 2:  # optimize away most common case.
            vf_pwnorm_fac **= (self.exponent - 1)

        inner_vf = vf.copy()

        for gi in inner_vf:
            gi *= pow(odl_abs(gi), self.exponent - 2)
            if self.exponent >= 2:
                # Any component that is zero is not divided with
                nz = (vf_pwnorm_fac.asarray() != 0)
                gi[nz] /= vf_pwnorm_fac[nz]
            else:
                # For exponents < 2 there will be a singularity if any
                # component is zero. This results in inf or nan. See the
                # documentation for further details.
                gi /= vf_pwnorm_fac

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
            given, its length must be equal to ``len(domain)``.
            By default, the weights are is taken from
            ``domain.weighting``. Note that this excludes unusual
            weightings with custom inner product, norm or dist.
        """
        if not isinstance(vfspace, ProductSpace):
            raise TypeError(f"`vfspace` {vfspace} is not a ProductSpace instance")
        if adjoint:
            super().__init__(
                domain=vfspace[0], range=vfspace, base_space=vfspace[0], linear=True
            )
        else:
            super().__init__(
                domain=vfspace, range=vfspace[0], base_space=vfspace[0], linear=True
            )

        # Bail out if the space is complex but we cannot take the complex
        # conjugate.
        if (vfspace.field == ComplexNumbers() and
                not hasattr(self.base_space.element_type, 'conj')):
            raise NotImplementedError(
                f"base space element type {self.base_space.element_type} does not implement conj() method required for complex inner products"
            )

        self._vecfield = vfspace.element(vecfield)

        # Handle weighting, including sanity checks
        if weighting is None:
            self.__is_weighted =  False
            if hasattr(vfspace.weighting, 'array'):
                self.__weights = vfspace.weighting.array
            elif hasattr(vfspace.weighting, 'const'):
                # Casting the constant to an array of constants is just bad
                self.__weights = [vfspace.weighting.const *vfspace[i].one()
                                  for i in range(len(vfspace))]
            else:
                raise ValueError(
                    f"weighting scheme {vfspace.weighting} of the domain does not define a weighting array or constant"
                )

        else:
            # Check if the input has already been sanitised, i.e is it an odl.Tensor
            if (isinstance(weighting, list)
                 and all(isinstance(w, Tensor) for w in weighting)):
                self.__weights = weighting
                self.__is_weighted = all(odl_all_equal(w, 1) for w in weighting)

            # these are required to provide an array-API compatible weighting parsing.
            else:
                if isinstance(weighting, (int, float)):
                    weighting = [weighting for i in range(len(vfspace))]

                weighted_flag = []
                for i in range(len(vfspace)):
                    if weighting[i] <= 0:
                        raise ValueError(
                            f"weighting array weighting contains invalid entry {weighting[i]}"
                        )
                    if weighting[i] in [1, 1.0]:
                        weighted_flag.append(False)
                    else:
                        weighted_flag.append(True)
                self.__is_weighted = bool(any(weighted_flag))

                weighting = [
                    vfspace[i].tspace.broadcast_to(weighting[i])
                    for i in range(len(vfspace))
                ]

                self.__weights = []
                for i in range(len(vfspace)):
                    self.weights.append(vfspace[i].element(weighting[i]))

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
        raise NotImplementedError("abstract method")


class PointwiseInner(PointwiseInnerBase):
    """Take the point-wise inner product with a given vector field.

    This operator takes the (weighted) inner product ::

        <F(x), G(x)> = sum_j ( w_j * F_j(x) * conj(G_j(x)) )

    for a given vector field ``G``, where ``F`` is the vector field
    acting as a variable to this operator.

    This implies that the `Operator.domain` is a power space of a
    discretized function space. For example, if ``X`` is a `DiscretizedSpace`
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
            given, its length must be equal to ``len(domain)``, and
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
        super().__init__(
            adjoint=False, vfspace=vfspace, vecfield=vecfield, weighting=weighting
        )

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

        if len(self.domain) == 1:
            return

        tmp = self.range.element()
        for vfi, gi, wi in zip(vf[1:], self.vecfield[1:], self.weights[1:]):

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

    The adjoint of the inner product operator is a mapping ::

        A^* : X --> X^d

    If the vector field space ``X^d`` is weighted by a vector ``v``,
    the adjoint, applied to a function ``h`` from ``X`` is the vector
    field ::

        x --> h(x) * (w / v) * G(x),

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
        vecfield : range `element-like`
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
                raise TypeError(f"`vfspace` {vfspace} is not a ProductSpace instance")
            if vfspace[0] != sspace:
                raise ValueError(
                    f"base space of the range is different from the given scalar space ({vfspace[0]} != {sspace})"
                )
        super().__init__(
            adjoint=True, vfspace=vfspace, vecfield=vecfield, weighting=weighting
        )

        # Get weighting from range
        if hasattr(self.range.weighting, 'array'):
            ### The tolist() is an ugly tweak to recover a list from the pspace weighting.array which is stored in numpy 
            self.__ran_weights = vfspace.element(self.range.weighting.array.tolist())
        elif hasattr(self.range.weighting, 'const'):
            # Casting the constant to an array of constants is just bad
            self.__ran_weights = [self.range.weighting.const *self.range[i].one()
                                  for i in range(len(self.range))]
        else:
            raise ValueError(
                f"weighting scheme {self.range.weighting} of the range does not define a weighting array or constant"
            )

    def _call(self, f, out):
        """Implement ``self(vf, out)``."""
        for vfi, oi, ran_wi, dom_wi in zip(self.vecfield, out,
                                           self.__ran_weights, self.weights):
            vfi.multiply(f, out=oi)
            # Removed the optimisation here, it would require casting ran_wi as odl.TensorSpaceElement
            # if not isclose(ran_wi, dom_wi).all():
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

    This operator takes the (weighted) sum ::

        sum(F(x)) = [ sum_j( w_j * F_j(x) ) ]

    where ``F`` is a vector field. This implies that
    the `Operator.domain` is a power space of a discretized function
    space. For example, if ``X`` is a `DiscretizedSpace` space, then
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
            given, its length must be equal to ``len(domain)``.
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
            raise TypeError(f"`vfspace` {vfspace} is not a ProductSpace instance")

        ones = vfspace.one()
        super().__init__(vfspace, vecfield=ones, weighting=weighting)


class MatrixOperator(Operator):
    """A matrix acting as a linear operator.

    This operator uses a matrix to represent an operator, and get its
    adjoint and inverse by doing computations on the matrix. This is in
    general a rather slow and memory-inefficient approach, and users are
    recommended to use other alternatives if possible.
    """

    def __init__(self, matrix, domain=None, range=None,
                 impl: Optional[str]=None,
                 device: Optional[str]=None,
                 axis=0):
        r"""Initialize a new instance.

        Parameters
        ----------
        matrix : `array-like` or `scipy.sparse.base.spmatrix`
            2-dimensional array representing the linear operator.
            For Scipy sparse matrices only tensor spaces with
            ``ndim == 1`` are allowed as ``domain``.
            The matrix is copied to `impl`/`device`, if these are
            specified (only once, when the operator is initialized).
            If a plain Python list is supplied, it will first
            be converted to a NumPy array.
        domain : `TensorSpace`, optional
            Space of elements on which the operator can act. Its
            ``dtype`` must be castable to ``range.dtype``.
            For the default ``None``, a space with 1 axis and size
            ``matrix.shape[1]`` is used, together with the matrix'
            data type.
        range : `TensorSpace`, optional
            Space of elements on to which the operator maps. Its
            ``shape`` and ``dtype`` attributes must match those
            of the result of the multiplication.
            For the default ``None``, the range is inferred from
            ``matrix``, ``domain`` and ``axis``.
        impl : `ArrayBackend`-identifying `str`, optional
            Which backend to use for the low-level matrix multiplication.
            If not explicitly provided, it will be inferred in the following
            order of preference, depending on what is available:
            1. from `domain`
            2. from `range`
            3. from `matrix`
        device : `str`, optional
            On which device to store the matrix.
            Same defaulting logic as for `impl`.
        axis : int, optional
            Sum over this axis of an input tensor in the
            multiplication.

        Examples
        --------
        By default, ``domain`` and ``range`` are spaces of with one axis:

        >>> m = np.ones((3, 4))
        >>> op = MatrixOperator(m)
        >>> op.domain
        rn(4)
        >>> op.range
        rn(3)
        >>> op([1, 2, 3, 4])
        rn(3).element([ 10.,  10.,  10.])

        For multi-dimensional arrays (tensors), the summation
        (contraction) can be performed along a specific axis. In
        this example, the number of matrix rows (4) must match the
        domain shape entry in the given axis:

        >>> dom = odl.rn((5, 4, 4))  # can use axis=1 or axis=2
        >>> op = MatrixOperator(m, domain=dom, axis=1)
        >>> op(dom.one()).shape
        (5, 3, 4)
        >>> op = MatrixOperator(m, domain=dom, axis=2)
        >>> op(dom.one()).shape
        (5, 4, 3)

        The operator also works on `uniform_discr` type spaces. Note,
        however, that the ``weighting`` of the domain is propagated to
        the range by default, in order to keep the correspondence between
        adjoint and transposed matrix:

        >>> space = odl.uniform_discr(0, 1, 4)
        >>> op = MatrixOperator(m, domain=space)
        >>> op(space.one())
        rn(3, weighting=0.25).element([ 4.,  4.,  4.])
        >>> np.array_equal(op.adjoint.matrix, m.T)
        True

        Notes
        -----
        For a matrix :math:`A \in \mathbb{F}^{n \times m}`, the
        operation on a tensor :math:`T \in \mathbb{F}^{n_1 \times
        \dots \times n_d}` is defined as the summation

        .. math::
            (A \cdot T)_{i_1, \dots, i_k, \dots, i_d} =
            \sum_{j=1}^m A_{i_k j} T_{i_1, \dots, j, \dots, i_d}.

        It produces a new tensor :math:`A \cdot T \in \mathbb{F}^{
        n_1 \times \dots \times n \times \dots \times n_d}`.
        """

        def infer_backend_from(default_backend):
            if impl is not None:
                self.__array_backend = lookup_array_backend(impl)
            else:
                assert isinstance(default_backend, ArrayBackend)
                self.__array_backend = default_backend

        def infer_device_from(default_device):
            self.__device = default_device if device is None else device

        self._sparse_format = lookup_sparse_format(matrix)

        if domain is not None:
            infer_backend_from(domain.array_backend)
            infer_device_from(domain.device)
            
        elif range is not None:
            infer_backend_from(range.array_backend)
            infer_device_from(range.device)

        elif self.is_sparse:
            if self._sparse_format.impl == 'scipy':
                infer_backend_from(lookup_array_backend('numpy'))
                infer_device_from('cpu')

            elif self._sparse_format.impl == 'pytorch':
                infer_backend_from(lookup_array_backend('pytorch'))
                infer_device_from(matrix.device)
                
            else:
                raise ValueError
        
        elif isinstance(matrix, (list, tuple)):
            infer_backend_from(lookup_array_backend('numpy'))
            infer_device_from('cpu')
        else:
            infer_backend_from(get_array_and_backend(matrix)[1])
            infer_device_from(matrix.device)

        self.__arr_ns = self.array_backend.array_namespace

        if self.is_sparse:
            if self._sparse_format.impl == 'scipy':
                if self.array_backend.impl != 'numpy':
                    raise TypeError(f"SciPy sparse matrices can only be used with NumPy on CPU, not {self.array_backend.impl}.")
                if self.device != 'cpu':
                    raise TypeError(f"SciPy sparse matrices can only be used with NumPy on CPU, not {device}.")
            elif self._sparse_format.impl == 'pytorch':
                if self.array_backend.impl != 'pytorch':
                    raise TypeError(f"PyTorch sparse matrices can only be used with Pytorch, not {self.array_backend.impl}.")
            self.__matrix = matrix

        elif isinstance(matrix, Tensor):
            self.__matrix = matrix.data
            self.__matrix = self.__arr_ns.asarray(
                matrix.data, device=self.__device, copy=AVOID_UNNECESSARY_COPY
            )
            while len(self.__matrix.shape) < 2:
                self.__matrix = self.__matrix[None]
        else:
            self.__matrix = self.__arr_ns.asarray(
                matrix, device=self.__device, copy=AVOID_UNNECESSARY_COPY
            )
            while len(self.__matrix.shape) < 2:
                self.__matrix = self.__matrix[None]

        self.__axis, axis_in = int(axis), axis
        if self.axis != axis_in:
            raise ValueError(f"`axis` must be integer, got {axis_in}")

        if self.matrix.ndim != 2:
            raise ValueError(f"`matrix` has {self.matrix.ndim} axes instead of 2")

        # Infer or check domain
        if domain is None:
            dtype = self.array_backend.identifier_of_dtype(self.matrix.dtype)
            domain = tensor_space((self.matrix.shape[1],),
                                  dtype=dtype,
                                  impl = self.array_backend.impl,
                                  device = self.device
                                  )
        else:
            if not isinstance(domain, TensorSpace):
                raise TypeError(
                    f"`domain` must be a `TensorSpace` instance, got {domain}"
                )

            if self.is_sparse and domain.ndim > 1:
                raise ValueError("`domain.ndim` > 1 unsupported for "
                                 "scipy sparse matrices")

            if domain.shape[axis] != self.matrix.shape[1]:
                raise ValueError(
                    f"`domain.shape[axis]` not equal to `matrix.shape[1]` ({domain.shape[axis]} != {self.matrix.shape[1]})"
                )

        range_shape = list(domain.shape)
        range_shape[self.axis] = self.matrix.shape[0]

        if range is None:
            # Infer range
            range_dtype = self.__arr_ns.result_type(
                self.matrix.dtype, domain.dtype)
            range_dtype = self.array_backend.identifier_of_dtype(range_dtype)
            if (range_shape != domain.shape and
                    isinstance(domain.weighting, ArrayWeighting)):
                # Cannot propagate weighting due to size mismatch.
                weighting = None
            else:
                weighting = domain.weighting
            range = tensor_space(range_shape, 
                                 impl = self.array_backend.impl,
                                 device=self.device,
                                 dtype=range_dtype,
                                 weighting=weighting,
                                 exponent=domain.exponent)
        else:
            # Check consistency of range
            if not isinstance(range, TensorSpace):
                raise TypeError(
                    f"`range` must be a `TensorSpace` instance, got {range}"
                )

            if range.shape != tuple(range_shape):
                raise ValueError(
                    f"expected `range.shape` = {tuple(range_shape)}, got {range.shape}"
                )

        # Check compatibility of data types
        result_dtype = self.__arr_ns.result_type(domain.dtype, self.matrix.dtype)
        if not can_cast(self.__arr_ns, result_dtype, range.dtype):
            raise ValueError(
                f"result data type {dtype_repr(result_dtype)} cannot be safely cast to range data type {dtype_repr(range.dtype)}"
            )

        super().__init__(domain, range, linear=True)

    @property
    def is_sparse(self) -> bool:
        """Whether the underlying matrix is stored in a format
        that optimized for mostly-zero entries.
        Note that this does not necessarily say anything about
        how sparse the matrix actually is."""
        return self._sparse_format is not None

    @property
    def matrix(self):
        """Matrix representing this operator."""
        return self.__matrix

    @property
    def array_backend(self):
        """Backend on which to carry out the BLAS matmul operation.
        Note that this does not necessarily have to be the same as
        either the range or domain of the operator, but by default it will
        be chosen such. If a different backend and/or device is used, the
        operator will always copy data to `self.array_backend` before
        carrying out the matrix multiplication, then copy the result to
        `self.range.array_backend`. Such copies should generally be avoided
        as they can be slow, but they can sometimes be justified if memory
        is scarce on one of the devices.
        """
        return self.__array_backend

    @property
    def device(self):
        """Computational device on which to carry out the BLAS operation.
        See remarks on `array_backend`."""
        return self.__device

    @property
    def axis(self):
        """Axis of domain elements over which is summed."""
        return self.__axis

    @property
    def adjoint(self):
        """Adjoint operator represented by the adjoint matrix.

        Returns
        -------
        adjoint : `MatrixOperator`
        """
        return MatrixOperator(self.matrix.conj().T,
                              domain=self.range, range=self.domain,
                              axis=self.axis)

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
        if self.is_sparse:
            matrix = self._sparse_format.to_dense(self.matrix)
        else:
            matrix = self.matrix
        return MatrixOperator(self.__arr_ns.linalg.inv(matrix),
                                domain=self.range, range=self.domain,
                                axis=self.axis, impl=self.domain.impl, device=self.domain.device)

    def _call(self, x):
        """Return ``self(x[, out])``."""

        if self.is_sparse:
            out = self._sparse_format.matmul_spmatrix_with_vector(self.matrix, x.data)
        else:
            dot = self.__arr_ns.tensordot(self.matrix, x.data, axes=([1], [self.axis]))
            # New axis ends up as first, need to swap it to its place
            out = self.__arr_ns.moveaxis(dot, 0, self.axis)

        return out

    def __repr__(self):
        """Return ``repr(self)``."""
        # Matrix printing itself in an executable way (for dense matrix)
        if self.is_sparse or self.array_backend.impl != 'numpy':
            matrix_str = repr(self.matrix)
        else:
            matrix_str = np.array2string(self.matrix, separator=', ')
        posargs = [matrix_str]

        # Optional arguments with defaults, inferred from the matrix
        range_shape = list(self.domain.shape)
        range_shape[self.axis] = self.matrix.shape[0]

        try:
            default_domain = tensor_space(self.matrix.shape[1],
                                          impl=self.array_backend.impl,
                                          dtype=self.matrix.dtype)
        except (ValueError, TypeError):
            default_domain = None
        try:
            default_range = tensor_space(range_shape,
                                         impl=self.array_backend.impl,
                                         dtype=self.matrix.dtype)
        except (ValueError, TypeError):
            default_range = None

        optargs = [
            ('domain', self.domain, default_domain),
            ('range', self.range, default_range),
            ('axis', self.axis, 0)
        ]

        inner_str = signature_string(
            posargs, optargs, sep=[", ", ", ", ",\n"], mod=[["!s"], ["!r", "!r", ""]]
        )
        return f"{self.__class__.__name__}(\n{indent(inner_str)}\n)"

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


def _normalize_sampling_points(sampling_points, ndim):
    """Normalize points to an ndim-long list of linear index arrays.

    This helper converts sampling indices for `SamplingOperator` from
    integers or array-like objects to a list of length ``ndim``, where
    each entry is a `numpy.ndarray` with ``dtype=int``.
    The function also checks if all arrays have equal lengths, and that
    they fulfill ``array.ndim=1`` (or ``size=0`` for if ``ndim == 0``).

    The result of this normalization is intended to be used for indexing
    an ``ndim``-dimensional array at ``sampling_points`` via NumPy fancy
    indexing, i.e., ``result = ndim_array[sampling_points]``.
    """
    sampling_points_in = sampling_points
    if ndim == 0:
        sampling_points = [np.array(sampling_points, dtype=int, copy=AVOID_UNNECESSARY_COPY)]
        if sampling_points[0].size != 0:
            raise ValueError("`sampling_points` must be empty for 0-dim. `domain`")
    elif ndim == 1:
        if isinstance(sampling_points, Integral):
            sampling_points = (sampling_points,)
        sampling_points = np.array(sampling_points, dtype=int, copy=AVOID_UNNECESSARY_COPY,
                                   ndmin=1)

        # Handle possible list of length one
        if sampling_points.ndim == 2 and sampling_points.shape[0] == 1:
            sampling_points = sampling_points[0]

        sampling_points = [sampling_points]
        if sampling_points[0].ndim > 1:
            raise ValueError(f"expected 1D index (array), got {sampling_points_in}")
    else:
        try:
            iter(sampling_points)
        except TypeError as exc:
            raise TypeError("`sampling_points` must be a sequence "
                            "for domain with ndim > 1") from exc
        else:
            if np.ndim(sampling_points) == 1:
                sampling_points = [np.array(p, dtype=int)
                                   for p in sampling_points]
            else:
                sampling_points = [
                    np.array(pts, dtype=int, copy=AVOID_UNNECESSARY_COPY, ndmin=1)
                    for pts in sampling_points]
                if any(pts.ndim != 1 for pts in sampling_points):
                    raise ValueError(
                        f"index arrays in `sampling_points` must be 1D, got {sampling_points_in}"
                    )

    return sampling_points


class SamplingOperator(Operator):

    """Operator that samples coefficients.

    The operator is defined by ::

        SamplingOperator(f) == c * f[sampling_points]

    with the weight ``c`` being determined by the variant. By choosing
    ``c = 1``, this operator approximates point evaluations or inner
    products with Dirac deltas, see option ``variant='point_eval'``.
    By choosing ``c = cell_volume``, it approximates the integration of
    ``f`` over the indexed cells, see option ``variant='integrate'``.
    """

    def __init__(self, domain, sampling_points, variant='point_eval'):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace`
            Set of elements on which this operator acts.
        sampling_points : 1D `array-like` or sequence of 1D array-likes
            Indices that determine the sampling points.
            In n dimensions, it should be a sequence of n arrays, where
            each member array is of equal length N. The indexed positions
            are ``(arr1[i], arr2[i], ..., arrn[i])``, in total N
            points.
            If ``domain`` is one-dimensional, a single array-like can be
            used. Likewise, a single point can be given as integer in 1D,
            and as a array-like sequence in nD.
        variant : {'point_eval', 'integrate'}, optional
            For ``'point_eval'`` this operator performs the sampling by
            evaluation the function at the sampling points. The
            ``'integrate'`` variant approximates integration by
            multiplying point evaluation with the cell volume.

        Examples
        --------
        Sampling in 1d can be done with a single index (an int) or a
        sequence of such:

        >>> space = odl.uniform_discr(0, 1, 4)
        >>> op = odl.SamplingOperator(space, sampling_points=1)
        >>> x = space.element([1, 2, 3, 4])
        >>> op(x)
        rn(1).element([ 2.])
        >>> op = odl.SamplingOperator(space, sampling_points=[1, 2, 1])
        >>> op(x)
        rn(3).element([ 2.,  3.,  2.])

        There are two variants ``'point_eval'`` (default) and
        ``'integrate'``, where the latter scales values by the cell
        volume to approximate the integral over the cells of the points:

        >>> op = odl.SamplingOperator(space, sampling_points=[1, 2, 1],
        ...                           variant='integrate')
        >>> space.cell_volume  # the scaling constant
        0.25
        >>> op(x)
        rn(3).element([ 0.5 ,  0.75,  0.5 ])

        In higher dimensions, a sequence of index array-likes must be
        given, or a single sequence for a single point:

        >>> space = odl.uniform_discr([0, 0], [1, 1], (2, 3))
        >>> # Sample at the index (0, 2)
        >>> op = odl.SamplingOperator(space, sampling_points=[0, 2])
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> op(x)
        rn(1).element([ 3.])
        >>> sampling_points = [[0, 1, 1],  # indices (0, 2), (1, 1), (1, 0)
        ...                    [2, 1, 0]]
        >>> op = odl.SamplingOperator(space, sampling_points)
        >>> op(x)
        rn(3).element([ 3.,  5.,  4.])
        """
        if not isinstance(domain, TensorSpace):
            raise TypeError(f"`domain` must be a `TensorSpace` instance, got {domain}")

        self.__sampling_points = _normalize_sampling_points(sampling_points,
                                                            domain.ndim)
        # Flatten indices during init for faster indexing later
        indices_flat = np.ravel_multi_index(self.sampling_points,
                                            dims=domain.shape)
        if np.isscalar(indices_flat):
            self._indices_flat = np.array([indices_flat], dtype=int)
        else:
            self._indices_flat = indices_flat
        self.__variant = str(variant).lower()
        if self.variant not in ("point_eval", "integrate"):
            raise ValueError(f"`variant` {variant} not understood")

        # Propagating the impl and device of the range
        ran = tensor_space(
            self.sampling_points[0].size,
            dtype=domain.dtype,
            impl=domain.impl,
            device=domain.device,
        )
        super().__init__(domain, ran, linear=True)

    @property
    def variant(self):
        """Weighting scheme for the sampling operator."""
        return self.__variant

    @property
    def sampling_points(self):
        """Indices where to sample the function."""
        return self.__sampling_points

    def _call(self, x):
        """Return values at indices, possibly weighted."""
        out = x.asarray().ravel()[self._indices_flat]

        if self.variant == 'point_eval':
            weights = 1.0
        elif self.variant == 'integrate':
            weights = getattr(self.domain, 'cell_volume', 1.0)
        else:
            raise RuntimeError(f"bad variant {self.variant}")

        if weights != 1.0:
            out *= weights

        return out

    @property
    def adjoint(self):
        """Adjoint of the sampling operator, a `WeightedSumSamplingOperator`.

        If each sampling point occurs only once, the adjoint consists
        in inserting the given values into the output at the sampling
        points. Duplicate sampling points are weighted with their
        multiplicity.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 3))
        >>> sampling_points = [[0, 1, 1, 0],
        ...                    [0, 1, 2, 0]]
        >>> op = odl.SamplingOperator(space, sampling_points)
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> np.abs(op.adjoint(op(x)).inner(x) - op(x).inner(op(x))) < 1e-10
        True

        The ``'integrate'`` variant adjoint puts ones at the indices in
        ``sampling_points``, multiplied by their multiplicity:

        >>> op = odl.SamplingOperator(space, sampling_points,
        ...                           variant='integrate')
        >>> op.adjoint(op.range.one())  # (0, 0) occurs twice
        uniform_discr([-1., -1.], [ 1.,  1.], (2, 3)).element(
            [[ 2.,  0.,  0.],
             [ 0.,  1.,  1.]]
        )
        >>> np.abs(op.adjoint(op(x)).inner(x) - op(x).inner(op(x))) < 1e-10
        True
        """
        if self.variant == 'point_eval':
            variant = 'dirac'
        elif self.variant == 'integrate':
            variant = 'char_fun'
        else:
            raise RuntimeError(f"bad variant {self.variant}")

        return WeightedSumSamplingOperator(self.domain, self.sampling_points, variant)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain, self.sampling_points]
        optargs = [('variant', self.variant, 'point_eval')]
        sig_str = signature_string(posargs, optargs, mod=['!r', ''],
                                   sep=[',\n', '', ',\n'])
        return f"{self.__class__.__name__}(\n{indent(sig_str)}\n)"

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class WeightedSumSamplingOperator(Operator):
    r"""Operator computing the sum of coefficients at sampling locations.

    This operator is the adjoint of `SamplingOperator`.

    Notes
    -----
    The weighted sum sampling operator for a sequence
    :math:`I = (i_n)_{n=1}^N`
    of indices (possibly with duplicates) is given by

    .. math::
        W_I(g)(x) = \sum_{i \in I} d_i(x) g_i,

    where :math:`g \in \mathbb{F}^N` is the value vector, and
    :math:`d_i` is either a Dirac delta or a characteristic function of
    the cell centered around the point indexed by :math:`i`.
    """

    def __init__(self, range, sampling_points, variant='char_fun'):
        """Initialize a new instance.

        Parameters
        ----------
        range : `TensorSpace`
            Set of elements into which this operator maps.
        sampling_points : 1D `array-like` or sequence of 1D array-likes
            Indices that determine the sampling points.
            In n dimensions, it should be a sequence of n arrays, where
            each member array is of equal length N. The indexed positions
            are ``(arr1[i], arr2[i], ..., arrn[i])``, in total N
            points.
            If ``range`` is one-dimensional, a single array-like can be
            used. Likewise, a single point can be given as integer in 1D,
            and as a array-like sequence in nD.
        variant : {'char_fun', 'dirac'}, optional
            This option determines which function to sum over.

        Examples
        --------
        In 1d, a single index (an int) or a sequence of such can be used
        for indexing.

        >>> space = odl.uniform_discr(0, 1, 4)
        >>> op = odl.WeightedSumSamplingOperator(space, sampling_points=1)
        >>> op.domain
        rn(1)
        >>> x = op.domain.element([1])
        >>> # Put value 1 at index 1
        >>> op(x)
        uniform_discr(0.0, 1.0, 4).element([ 0.,  1.,  0.,  0.])
        >>> op = odl.WeightedSumSamplingOperator(space,
        ...                                      sampling_points=[1, 2, 1])
        >>> op.domain
        rn(3)
        >>> x = op.domain.element([1, 0.5, 0.25])
        >>> # Index 1 occurs twice and gets two contributions (1 and 0.25)
        >>> op(x)
        uniform_discr(0.0, 1.0, 4).element([ 0.  ,  1.25,  0.5 ,  0.  ])

        The ``'dirac'`` variant scales the values by the reciprocal
        cell volume of the operator range:

        >>> op = odl.WeightedSumSamplingOperator(
        ...     space, sampling_points=[1, 2, 1], variant='dirac')
        >>> x = op.domain.element([1, 0.5, 0.25])
        >>> 1 / op.range.cell_volume  # the scaling constant
        4.0
        >>> op(x)
        uniform_discr(0.0, 1.0, 4).element([ 0.,  5.,  2.,  0.])

        In higher dimensions, a sequence of index array-likes must be
        given, or a single sequence for a single point:

        >>> space = odl.uniform_discr([0, 0], [1, 1], (2, 3))
        >>> # Sample at the index (0, 2)
        >>> op = odl.WeightedSumSamplingOperator(space,
        ...                                      sampling_points=[0, 2])
        >>> x = op.domain.element([1])
        >>> # Insert the value 1 at index (0, 2)
        >>> op(x)
        uniform_discr([ 0.,  0.], [ 1.,  1.], (2, 3)).element(
            [[ 0.,  0.,  1.],
             [ 0.,  0.,  0.]]
        )
        >>> sampling_points = [[0, 1],  # indices (0, 2) and (1, 1)
        ...                    [2, 1]]
        >>> op = odl.WeightedSumSamplingOperator(space, sampling_points)
        >>> x = op.domain.element([1, 2])
        >>> op(x)
        uniform_discr([ 0.,  0.], [ 1.,  1.], (2, 3)).element(
            [[ 0.,  0.,  1.],
             [ 0.,  2.,  0.]]
        )
        """
        if not isinstance(range, TensorSpace):
            raise TypeError(f"`range` must be a `TensorSpace` instance, got {range}")
        self.__sampling_points = _normalize_sampling_points(sampling_points, range.ndim)
        # Convert a list of index arrays to linear index array
        indices_flat = np.ravel_multi_index(self.sampling_points, dims=range.shape)
        if np.isscalar(indices_flat):
            indices_flat = np.array([indices_flat], dtype=int)
        else:
            indices_flat = np.array(indices_flat, dtype=int)

        ### Always converting the indices to the right data type
        self._indices_flat = range.array_backend.array_constructor(
            indices_flat, dtype=int, device=range.device
        )

        self.__variant = str(variant).lower()
        if self.variant not in ('dirac', 'char_fun'):
            raise ValueError(f"`variant` {variant} not understood")

        # Recording the namespace for bincount
        self.namespace = range.array_backend.array_namespace

        # Propagating the impl and device of the range
        domain = tensor_space(
            self.sampling_points[0].size,
            dtype=range.dtype,
            impl=range.impl,
            device=range.device,
        )
        super().__init__(domain, range, linear=True)

    @property
    def variant(self):
        """Weighting scheme for the operator."""
        return self.__variant

    @property
    def sampling_points(self):
        """Indices where to sample the function."""
        return self.__sampling_points

    def _call(self, x):
        """Sum all values if indices are given multiple times."""
        y = self.namespace.bincount(self._indices_flat, weights=x.data,
                        minlength=self.range.size)

        out = y.reshape(self.range.shape)

        if self.variant == 'dirac':
            weights = getattr(self.range, 'cell_volume', 1.0)
        elif self.variant == 'char_fun':
            weights = 1.0
        else:
            raise RuntimeError(f'The variant "{self.variant}" is not yet supported')

        if weights != 1.0:
            out /= weights

        return out

    @property
    def adjoint(self):
        """Adjoint of this operator, a `SamplingOperator`.

        The ``'char_fun'`` variant of this operator corresponds to the
        ``'integrate'`` sampling operator, and ``'dirac'`` corresponds to
        ``'point_eval'``.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 3))
        >>> # Point (0, 0) occurs twice
        >>> sampling_points = [[0, 1, 1, 0],
        ...                    [0, 1, 2, 0]]
        >>> op = odl.WeightedSumSamplingOperator(space, sampling_points,
        ...                                      variant='dirac')
        >>> y = op.range.element([[1, 2, 3],
        ...                       [4, 5, 6]])
        >>> op.adjoint(y)
        rn(4).element([ 1.,  5.,  6.,  1.])
        >>> x = op.domain.element([1, 2, 3, 4])
        >>> np.abs(op.adjoint(op(x)).inner(x) - op(x).inner(op(x))) < 1e-10
        True
        >>> op = odl.WeightedSumSamplingOperator(space, sampling_points,
        ...                                      variant='char_fun')
        >>> np.abs(op.adjoint(op(x)).inner(x) - op(x).inner(op(x))) < 1e-10
        True
        """
        if self.variant == 'dirac':
            variant = 'point_eval'
        elif self.variant == 'char_fun':
            variant = 'integrate'
        else:
            raise RuntimeError(f'The variant "{self.variant}" is not yet supported')

        return SamplingOperator(self.range, self.sampling_points, variant)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.range, self.sampling_points]
        optargs = [('variant', self.variant, 'char_fun')]
        sig_str = signature_string(posargs, optargs, mod=['!r', ''],
                                   sep=[',\n', '', ',\n'])
        return f"{self.__class__.__name__}(\n{indent(sig_str)}\n)"

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class FlatteningOperator(Operator):
    """Operator that reshapes the object as a column vector.

    The operation performed by this operator is ::

        FlatteningOperator(x) == ravel(x)

    The range of this operator is always a `TensorSpace`, i.e., even if
    the domain is a discrete function space.
    """

    def __init__(self, domain):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `TensorSpace`
            Set of elements on which this operator acts.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 3))
        >>> op = odl.FlatteningOperator(space)
        >>> op.range
        rn(6)
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> op(x)
        rn(6).element([ 1.,  2.,  3.,  4.,  5.,  6.])
        """
        if not isinstance(domain, TensorSpace):
            raise TypeError(f"`domain` must be a `TensorSpace` instance, got {domain}")

        range = tensor_space(domain.size, dtype=domain.dtype)
        super().__init__(domain, range, linear=True)

    def _call(self, x):
        """Flatten ``x``."""
        return self.range.element(x.data.reshape([self.range.shape[0]]))

    @property
    def adjoint(self):
        """Adjoint of the flattening, a scaled version of the `inverse`.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 4))
        >>> op = odl.FlatteningOperator(space)
        >>> y = op.range.element([1, 2, 3, 4, 5, 6, 7, 8])
        >>> 1 / space.cell_volume  # the scaling factor
        2.0
        >>> op.adjoint(y)
        uniform_discr([-1., -1.], [ 1.,  1.], (2, 4)).element(
            [[  2.,   4.,   6.,   8.],
             [ 10.,  12.,  14.,  16.]]
        )
        >>> x = space.element([[1, 2, 3, 4],
        ...                    [5, 6, 7, 8]])
        >>> np.abs(op.adjoint(op(x)).inner(x) - op(x).inner(op(x))) < 1e-10
        True
        """
        scaling = getattr(self.domain, 'cell_volume', 1.0)
        return 1 / scaling * self.inverse

    @property
    def inverse(self):
        """Operator that reshapes to original shape.

        Examples
        --------
        >>> space = odl.uniform_discr([-1, -1], [1, 1], shape=(2, 4))
        >>> op = odl.FlatteningOperator(space)
        >>> y = op.range.element([1, 2, 3, 4, 5, 6, 7, 8])
        >>> op.inverse(y)
        uniform_discr([-1., -1.], [ 1.,  1.], (2, 4)).element(
            [[ 1.,  2.,  3.,  4.],
             [ 5.,  6.,  7.,  8.]]
        )
        >>> op(op.inverse(y)) == y
        True
        """
        op = self
        scaling = getattr(self.domain, 'cell_volume', 1.0)

        class FlatteningOperatorInverse(Operator):
            """Inverse of `FlatteningOperator`.

            This operator reshapes a flat vector back to original shape::

                FlatteningOperatorInverse(x) == reshape(x, orig_shape)
            """

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(op.range, op.domain, linear=True)

            def _call(self, x):
                """Reshape ``x`` back to n-dim. shape."""
                return np.reshape(x.asarray(), self.range.shape)

            @property
            def adjoint(self):
                """Adjoint of this operator, a scaled `FlatteningOperator`."""
                return scaling * op

            @property
            def inverse(self):
                """Inverse of this operator."""
                return op

            def __repr__(self):
                """Return ``repr(self)``."""
                return f"{op}.inverse"

            def __str__(self):
                """Return ``str(self)``."""
                return repr(self)

        return FlatteningOperatorInverse()

    def __repr__(self):
        """Return ``repr(self)``."""
        return f"{self.__class__.__name__}({self.domain})"

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
    from odl.core.util.testutils import run_doctests
    run_doctests()
