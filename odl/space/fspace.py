# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Spaces of functions with common domain and range."""

from __future__ import print_function, division, absolute_import
from builtins import object
from inspect import isfunction
import numpy as np

from odl.operator.operator import _dispatch_call_args
from odl.set import (RealNumbers, ComplexNumbers, Set, LinearSpace,
                     LinearSpaceElement)
from odl.util import (
    is_real_dtype, is_complex_floating_dtype, dtype_str,
    complex_dtype, real_dtype,
    is_valid_input_array, is_valid_input_meshgrid,
    out_shape_from_array, out_shape_from_meshgrid, vectorize,
    signature_string)
from odl.util.utility import preload_first_arg


__all__ = ('FunctionSpace', 'FunctionSpaceElement')


def _default_in_place(func, x, out, **kwargs):
    """Default in-place evaluation method."""
    out[:] = func(x, **kwargs)
    return out


def _default_out_of_place(func, x, **kwargs):
    """Default in-place evaluation method."""
    if is_valid_input_array(x, func.domain.ndim):
        out_shape = out_shape_from_array(x)
    elif is_valid_input_meshgrid(x, func.domain.ndim):
        out_shape = out_shape_from_meshgrid(x)
    else:
        raise TypeError('cannot use in-place method to implement '
                        'out-of-place non-vectorized evaluation')

    dtype = func.space.out_dtype
    if dtype is None:
        dtype = np.result_type(*x)

    out = np.empty(out_shape, dtype=dtype)
    func(x, out=out, **kwargs)
    return out


def _broadcast_to(array, shape):
    """Wrapper for the numpy function broadcast_to.

    Added since we dont require numpy 1.10 and hence cant guarantee that this
    exists.
    """
    array = np.asarray(array)
    try:
        return np.broadcast_to(array, shape)
    except AttributeError:
        # The above requires numpy 1.10, fallback impl else
        shape = [m if n == 1 and m != 1 else 1
                 for n, m in zip(array.shape, shape)]
        return array + np.zeros(shape, dtype=array.dtype)


class FunctionSpace(LinearSpace):

    """A vector space of functions."""

    def __init__(self, domain, out_dtype=float):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `Set`
            The domain of the functions.
        out_dtype : optional
            Scalar data type of the return value of a function in this
            space. Can be provided in any way the `numpy.dtype`
            constructor understands, e.g. as built-in type or as a string.
            For ``None``, dtypes of results are inferred lazily at runtime.

        Examples
        --------
        >>> domain = odl.IntervalProd(0, 1)
        >>> odl.FunctionSpace(domain)
        FunctionSpace(IntervalProd(0.0, 1.0))
        >>> odl.FunctionSpace(domain, out_dtype='float32')
        FunctionSpace(IntervalProd(0.0, 1.0), out_dtype='float32')
        """
        if not isinstance(domain, Set):
            raise TypeError('`domain` {!r} not a `Set` instance'
                            ''.format(domain))

        if is_real_dtype(out_dtype):
            field = RealNumbers()
        elif is_complex_floating_dtype(out_dtype):
            field = ComplexNumbers()
        else:
            field = None

        super(FunctionSpace, self).__init__(field)
        self.__domain = domain
        self.__out_dtype = None if out_dtype is None else np.dtype(out_dtype)

        # Init cache attributes for real / complex variants
        if self.field == RealNumbers():
            self.__real_out_dtype = self.out_dtype
            self.__real_space = self
            self.__complex_out_dtype = complex_dtype(self.out_dtype,
                                                     default=np.dtype(object))
            self.__complex_space = None
        elif self.field == ComplexNumbers():
            self.__real_out_dtype = real_dtype(self.out_dtype)
            self.__real_space = None
            self.__complex_out_dtype = self.out_dtype
            self.__complex_space = self
        else:
            self.__real_out_dtype = None
            self.__real_space = None
            self.__complex_out_dtype = None
            self.__complex_space = None

    @property
    def domain(self):
        """Common domain of all functions in this set."""
        return self.__domain

    @property
    def out_dtype(self):
        """Output data type of this function.

        If ``None``, the output data type is not uniquely pre-defined.
        """
        return self.__out_dtype

    @property
    def real_out_dtype(self):
        """The real dtype corresponding to this space's `out_dtype`."""
        return self.__real_out_dtype

    @property
    def complex_out_dtype(self):
        """The complex dtype corresponding to this space's `out_dtype`."""
        return self.__complex_out_dtype

    @property
    def real_space(self):
        """The space corresponding to this space's `real_dtype`."""
        return self.astype(self.real_out_dtype)

    @property
    def complex_space(self):
        """The space corresponding to this space's `complex_dtype`."""
        return self.astype(self.complex_out_dtype)

    def element(self, fcall=None, vectorized=True):
        """Create a `FunctionSpace` element.

        Parameters
        ----------
        fcall : callable, optional
            The actual instruction for out-of-place evaluation.
            It must return a `FunctionSpace.range` element or a
            `numpy.ndarray` of such (vectorized call).
            If ``fcall`` is a `FunctionSpaceElement`, it is wrapped
            as a new `FunctionSpaceElement`.
            For ``None``, the `zero` element is returned.
        vectorized : bool, optional
            Whether ``fcall`` supports vectorized evaluation.

        Returns
        -------
        element : `FunctionSpaceElement`
            The new element, always supports vectorization

        Notes
        -----
        For ``vectorized=False``, the function is decorated with a
        vectorizer, which makes two elements created this way from the same
        function being regarded as *not equal*.

        See Also
        --------
        odl.discr.grid.RectGrid.meshgrid : efficient grids for function
            evaluation
        """
        if fcall is None:
            return self.zero()
        elif fcall in self:
            return fcall
        elif not callable(fcall):
            raise TypeError('`fcall` {!r} is not callable'.format(fcall))
        else:
            if not vectorized:
                if self.field == RealNumbers():
                    otypes = ['float64']
                elif self.field == ComplexNumbers():
                    otypes = ['complex128']
                else:
                    otypes = []

                fcall = vectorize(otypes=otypes)(fcall)

            return self.element_type(self, fcall)

    def zero(self):
        """Function mapping everything to zero.

        This function is the additive unit in the function space.

        Since `FunctionSpace.lincomb` may be slow, we implement this function
        directly.
        """
        def zero_vec(x, out=None):
            """Zero function, vectorized."""
            if is_valid_input_meshgrid(x, self.domain.ndim):
                out_shape = out_shape_from_meshgrid(x)
            elif is_valid_input_array(x, self.domain.ndim):
                out_shape = out_shape_from_array(x)
            else:
                raise TypeError('invalid input type')

            if out is None:
                return np.zeros(out_shape, dtype=self.out_dtype)
            else:
                # Need to go through an array to fill with the correct
                # zero value for all dtypes
                fill_value = np.zeros(1, dtype=self.out_dtype)[0]
                out.fill(fill_value)

        return self.element_type(self, zero_vec)

    def one(self):
        """Function mapping everything to one.

        This function is the multiplicative unit in the function space.
        """
        def one_vec(x, out=None):
            """One function, vectorized."""
            if is_valid_input_meshgrid(x, self.domain.ndim):
                out_shape = out_shape_from_meshgrid(x)
            elif is_valid_input_array(x, self.domain.ndim):
                out_shape = out_shape_from_array(x)
            else:
                raise TypeError('invalid input type')

            if out is None:
                return np.ones(out_shape, dtype=self.out_dtype)
            else:
                # Need to go through an array to fill with the correct
                # one value for all dtypes
                fill_value = np.ones(1, dtype=self.out_dtype)[0]
                out.fill(fill_value)

        return self.element_type(self, one_vec)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `FunctionSpace` with same
            `FunctionSpace.domain` and `FunctionSpace.range`, ``False``
            otherwise.
        """
        if other is self:
            return True

        return (type(self) is type(other) and
                self.domain == other.domain and
                self.out_dtype == other.out_dtype)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.domain, self.out_dtype))

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `FunctionSpaceElement`
            whose `FunctionSpaceElement.space` attribute
            equals this space, ``False`` otherwise.
        """
        return (isinstance(other, self.element_type) and
                other.space == self)

    def _astype(self, out_dtype):
        """Internal helper for ``astype``."""
        return type(self)(self.domain, out_dtype=out_dtype)

    def astype(self, out_dtype):
        """Return a copy of this space with new ``out_dtype``.

        Parameters
        ----------
        out_dtype :
            Output data type of the returned space. Can be given in any
            way `numpy.dtype` understands, e.g. as string ('complex64')
            or data type (complex). None is interpreted as 'float64'.

        Returns
        -------
        newspace : `FunctionSpace`
            The version of this space with given data type
        """
        out_dtype = np.dtype(out_dtype)
        if out_dtype == self.out_dtype:
            return self

        # Caching for real and complex versions (exact dtyoe mappings)
        if out_dtype == self.real_out_dtype:
            if self.__real_space is None:
                self.__real_space = self._astype(out_dtype)
            return self.__real_space
        elif out_dtype == self.complex_out_dtype:
            if self.__complex_space is None:
                self.__complex_space = self._astype(out_dtype)
            return self.__complex_space
        else:
            return self._astype(out_dtype)

    def _lincomb(self, a, x1, b, x2, out):
        """Raw linear combination of ``x1`` and ``x2``.

        Notes
        -----
        The additions and multiplications are implemented via simple
        Python functions, so non-vectorized versions are slow.
        """
        # Store to allow aliasing
        x1_call_oop = x1._call_out_of_place
        x1_call_ip = x1._call_in_place
        x2_call_oop = x2._call_out_of_place
        x2_call_ip = x2._call_in_place

        def lincomb_call_out_of_place(x):
            """Linear combination, out-of-place version."""
            # Due to vectorization, at least one call must be made to
            # ensure the correct final shape. The rest is optimized as
            # far as possible.
            if a == 0 and b != 0:
                out = np.asarray(x2_call_oop(x), dtype=self.out_dtype)
                if b != 1:
                    out *= b
            elif b == 0:  # Contains the case a == 0
                out = np.asarray(x1_call_oop(x), dtype=self.out_dtype)
                if a != 1:
                    out *= a
            else:
                out = np.asarray(x1_call_oop(x), dtype=self.out_dtype)
                if a != 1:
                    out *= a
                tmp = np.asarray(x2_call_oop(x), dtype=self.out_dtype)
                if b != 1:
                    tmp *= b
                out += tmp
            return out

        def lincomb_call_in_place(x, out):
            """Linear combination, in-place version."""
            if not isinstance(out, np.ndarray):
                raise TypeError('in-place evaluation only possible if output '
                                'is of type `numpy.ndarray`')
            # TODO: this could be optimized for the case when x1 and x2
            # are identical
            if a == 0 and b == 0:
                out *= 0
            elif a == 0 and b != 0:
                x2_call_ip(x, out)
                if b != 1:
                    out *= b
            elif b == 0 and a != 0:
                x1_call_ip(x, out)
                if a != 1:
                    out *= a
            else:
                tmp = np.empty_like(out)
                x1_call_ip(x, out)
                x2_call_ip(x, tmp)
                if a != 1:
                    out *= a
                if b != 1:
                    tmp *= b
                out += tmp
            return out

        out._call_out_of_place = lincomb_call_out_of_place
        out._call_in_place = lincomb_call_in_place
        out._call_has_out = out._call_out_optional = True
        return out

    def _multiply(self, x1, x2, out):
        """Raw pointwise multiplication of two functions.

        Notes
        -----
        The multiplication is implemented with a simple Python
        function, so the non-vectorized versions are slow.
        """
        # Store to allow aliasing
        x1_call_oop = x1._call_out_of_place
        x1_call_ip = x1._call_in_place
        x2_call_oop = x2._call_out_of_place
        x2_call_ip = x2._call_in_place

        def product_call_out_of_place(x):
            """Product out-of-place evaluation function."""
            return np.asarray(x1_call_oop(x) * x2_call_oop(x),
                              dtype=self.out_dtype)

        def product_call_in_place(x, out):
            """Product in-place evaluation function."""
            tmp = np.empty_like(out, dtype=self.out_dtype)
            x1_call_ip(x, out)
            x2_call_ip(x, tmp)
            out *= tmp
            return out

        out._call_out_of_place = product_call_out_of_place
        out._call_in_place = product_call_in_place
        out._call_has_out = out._call_out_optional = True
        return out

    def _divide(self, x1, x2, out):
        """Raw pointwise division of two functions."""
        # Store to allow aliasing
        x1_call_oop = x1._call_out_of_place
        x1_call_ip = x1._call_in_place
        x2_call_oop = x2._call_out_of_place
        x2_call_ip = x2._call_in_place

        def quotient_call_out_of_place(x):
            """Quotient out-of-place evaluation function."""
            return np.asarray(x1_call_oop(x) / x2_call_oop(x),
                              dtype=self.out_dtype)

        def quotient_call_in_place(x, out):
            """Quotient in-place evaluation function."""
            tmp = np.empty_like(out, dtype=self.out_dtype)
            x1_call_ip(x, out)
            x2_call_ip(x, tmp)
            out /= tmp
            return out

        out._call_out_of_place = quotient_call_out_of_place
        out._call_in_place = quotient_call_in_place
        out._call_has_out = out._call_out_optional = True
        return out

    def _scalar_power(self, x, p, out):
        """Raw p-th power of a function, p integer or general scalar."""
        x_call_oop = x._call_out_of_place
        x_call_ip = x._call_in_place

        def pow_posint(x, n):
            """Recursion to calculate the n-th power out-of-place."""
            if isinstance(x, np.ndarray):
                y = x.copy()
                return ipow_posint(y, n)
            else:
                return x ** n

        def ipow_posint(x, n):
            """Recursion to calculate the n-th power in-place."""
            if n == 1:
                return x
            elif n % 2 == 0:
                x *= x
                return ipow_posint(x, n // 2)
            else:
                tmp = x.copy()
                x *= x
                ipow_posint(x, n // 2)
                x *= tmp
                return x

        def power_call_out_of_place(x):
            """Power out-of-place evaluation function."""
            if p == 0:
                return self.one()
            elif p == int(p) and p >= 1:
                return np.asarray(pow_posint(x_call_oop(x), int(p)),
                                  dtype=self.out_dtype)
            else:
                return np.power(x_call_oop(x), p).astype(self.out_dtype)

        def power_call_in_place(x, out):
            """Power in-place evaluation function."""
            if p == 0:
                out.assign(self.one())

            x_call_ip(x, out)
            if p == int(p) and p >= 1:
                return ipow_posint(out, int(p))
            else:
                out **= p
                return out

        out._call_out_of_place = power_call_out_of_place
        out._call_in_place = power_call_in_place
        out._call_has_out = out._call_out_optional = True
        return out

    def _realpart(self, x):
        """Function returning the real part of a result."""
        x_call_oop = x._call_out_of_place

        def realpart_oop(x):
            return np.asarray(x_call_oop(x), dtype=self.out_dtype).real

        if is_real_dtype(self.out_dtype):
            return x
        else:
            rdtype = real_dtype(self.out_dtype)
            rspace = self.astype(rdtype)
            return rspace.element(realpart_oop)

    def _imagpart(self, x):
        """Function returning the imaginary part of a result."""
        x_call_oop = x._call_out_of_place

        def imagpart_oop(x):
            return np.asarray(x_call_oop(x), dtype=self.out_dtype).imag

        if is_real_dtype(self.out_dtype):
            return self.zero()
        else:
            rdtype = real_dtype(self.out_dtype)
            rspace = self.astype(rdtype)
            return rspace.element(imagpart_oop)

    def _conj(self, x):
        """Function returning the complex conjugate of a result."""
        x_call_oop = x._call_out_of_place

        def conj_oop(x):
            return np.asarray(x_call_oop(x), dtype=self.out_dtype).conj()

        if is_real_dtype(self.out_dtype):
            return x
        else:
            return self.element(conj_oop)

    @property
    def examples(self):
        """Return example functions in the space.

        Example functions include:

        Zero
        One
        Heaviside function
        Hypercube characteristic function
        Hypersphere characteristic function
        Gaussian
        Linear gradients
        """
        # Get the points and calculate some statistics on them
        mins = self.domain.min()
        maxs = self.domain.max()
        means = (maxs + mins) / 2.0
        stds = (maxs - mins) / 4.0
        ndim = getattr(self.domain, 'ndim', None)

        # Zero and One
        yield ('Zero', self.zero())
        yield ('One', self.one())

        # Indicator function in first dimension
        def _step_fun(x):
            if ndim == 1:
                return x > means[0]
            else:
                return (x[0] > means[0]) + 0 * sum(x[1:])  # fix size

        yield ('Step', self.element(_step_fun))

        # Indicator function on hypercube
        def _cube_fun(x):
            if ndim > 1:
                result = True
                for points, mean, std in zip(x, means, stds):
                    result = np.logical_and(result, points < mean + std)
                    result = np.logical_and(result, points > mean - std)
            else:
                result = np.logical_and(x < means + stds,
                                        x > means - stds)

            return result

        yield ('Cube', self.element(_cube_fun))

        # Indicator function on hypersphere
        if self.domain.ndim > 1:  # Only if ndim > 1, don't duplicate cube
            def _sphere_fun(x):
                if ndim == 1:
                    x = (x,)

                r = 0

                for points, mean, std in zip(x, means, stds):
                    r = r + (points - mean) ** 2 / std ** 2

                return r < 1.0

            yield ('Sphere', self.element(_sphere_fun))

        # Gaussian function
        def _gaussian_fun(x):
            if ndim == 1:
                x = (x,)

            r2 = 0
            for points, mean, std in zip(x, means, stds):
                r2 = r2 + (points - mean) ** 2 / ((std / 2) ** 2)

            return np.exp(-r2)

        yield ('Gaussian', self.element(_gaussian_fun))

        # Gradient in each dimensions
        for dim in range(self.domain.ndim):
            def _gradient_fun(x):
                if ndim == 1:
                    x = (x,)

                s = 0
                for ind in range(len(x)):
                    if ind == dim:
                        s = s + (x[ind] - mins[ind]) / (maxs[ind] - mins[ind])
                    else:
                        s = s + x[ind] * 0  # Correct broadcast size

                return s

            yield ('grad {}'.format(dim), self.element(_gradient_fun))

        # Gradient in all dimensions
        if self.domain.ndim > 1:  # Only if ndim > 1, don't duplicate grad 0
            def _all_gradient_fun(x):
                if ndim == 1:
                    x = (x,)

                s = 0
                for ind in range(len(x)):
                    s = s + (x[ind] - mins[ind]) / (maxs[ind] - mins[ind])

                return s

            yield ('Grad all', self.element(_all_gradient_fun))

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = [('out_dtype', dtype_str(self.out_dtype), 'float')]
        inner_str = signature_string(posargs, optargs, mod=['!r', ''])
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

    @property
    def element_type(self):
        """`FunctionSpaceElement`"""
        return FunctionSpaceElement


class FunctionSpaceElement(LinearSpaceElement):

    """Representation of a `FunctionSpace` element."""

    def __init__(self, fspace, fcall):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            Set of functions this element lives in.
        fcall : callable
            The actual instruction for out-of-place evaluation.
            It must return a `FunctionSpace.range` element or a
            `numpy.ndarray` of such (vectorized call).

        Examples
        --------
        >>> fspace = odl.FunctionSpace(odl.IntervalProd(0, 1))
        >>> sin = fspace.element(np.sin)
        >>> sin
        FunctionSpace(IntervalProd(0.0, 1.0)).element(<ufunc 'sin'>)
        >>> print(sin)
        sin: [0.0, 1.0] --> float64
        """
        super(FunctionSpaceElement, self).__init__(fspace)

        # Determine which type of implementation fcall is
        if isinstance(fcall, FunctionSpaceElement):
            call_has_out, call_out_optional, _ = _dispatch_call_args(
                bound_call=fcall._call)
        elif hasattr(fcall, 'nin') and hasattr(fcall, 'nout'):
            # Numpy Ufuncs and similar objects (e.g. Numba DUfuncs)
            fname = getattr(fcall, '__name__', str(fcall))
            if fcall.nin != 1:
                raise ValueError('ufunc {} has {} input parameter(s), '
                                 'expected 1'.format(fname, fcall.nin))
            if fcall.nout != 1:
                raise ValueError('ufunc {} has {} output parameter(s), '
                                 'expected 1'.format(fname, fcall.nout))
            call_has_out = call_out_optional = True
        elif isfunction(fcall):
            call_has_out, call_out_optional, _ = _dispatch_call_args(
                unbound_call=fcall)
        elif callable(fcall):
            call_has_out, call_out_optional, _ = _dispatch_call_args(
                bound_call=fcall.__call__)
        else:
            raise TypeError('type {!r} not callable')

        self._call_has_out = call_has_out
        self._call_out_optional = call_out_optional

        if not call_has_out:
            # Out-of-place-only
            self._call_in_place = preload_first_arg(self, 'in-place')(
                _default_in_place)
            self._call_out_of_place = fcall
        elif call_out_optional:
            # Dual-use
            self._call_in_place = self._call_out_of_place = fcall
        else:
            # In-place-only
            self._call_in_place = fcall
            self._call_out_of_place = preload_first_arg(self, 'out-of-place')(
                _default_out_of_place)

    @property
    def domain(self):
        """Set of objects on which this function can be evaluated."""
        return self.space.domain

    @property
    def out_dtype(self):
        """Output data type of this function.
        If ``None``, the output data type is not uniquely pre-defined.
        """
        return self.space.out_dtype

    def _call(self, x, out=None, **kwargs):
        """Raw evaluation method."""
        if out is None:
            return self._call_out_of_place(x, **kwargs)
        else:
            self._call_in_place(x, out=out, **kwargs)

    def __call__(self, x, out=None, **kwargs):
        """Return ``self(x[, out, **kwargs])``.

        Parameters
        ----------
        x : `domain` `element-like`, `meshgrid` or `numpy.ndarray`
            Input argument for the function evaluation. Conditions
            on ``x`` depend on its type:

            element-like: must be a castable to a domain element

            meshgrid: length must be ``space.ndim``, and the arrays must
            be broadcastable against each other.

            array:  shape must be ``(d, N)``, where ``d`` is the number
            of dimensions of the function domain

        out : `numpy.ndarray`, optional
            Output argument holding the result of the function
            evaluation, can only be used for vectorized
            functions. Its shape must be equal to
            ``np.broadcast(*x).shape``.

        Other Parameters
        ----------------
        bounds_check : bool, optional
            If ``True``, check if all input points lie in the function
            domain in the case of vectorized evaluation. This requires
            the domain to implement `Set.contains_all`.
            Default: ``True`` if `space` has a ``field``, ``False``
            otherwise.

        Returns
        -------
        out : range element or array of elements
            Result of the function evaluation. If ``out`` was provided,
            the returned object is a reference to it.

        Raises
        ------
        TypeError
            If ``x`` is not a valid vectorized evaluation argument

            If ``out`` is not a range element or a `numpy.ndarray`
            of range elements

        ValueError
            If evaluation points fall outside the valid domain
        """
        bounds_check = kwargs.pop('bounds_check', self.space.field is not None)
        if bounds_check and not hasattr(self.domain, 'contains_all'):
            raise AttributeError('bounds check not possible for '
                                 'domain {}, missing `contains_all()` '
                                 'method'.format(self.domain))

        if bounds_check and not hasattr(self.space.field, 'contains_all'):
            raise AttributeError('bounds check not possible for '
                                 'range {}, missing `contains_all()` '
                                 'method'.format(self.space.field))

        ndim = getattr(self.domain, 'ndim', None)
        # Check for input type and determine output shape
        if is_valid_input_meshgrid(x, ndim):
            out_shape = out_shape_from_meshgrid(x)
            scalar_out = False
            # Avoid operations on tuples like x * 2 by casting to array
            if ndim == 1:
                x = x[0][None, ...]
        elif is_valid_input_array(x, ndim):
            x = np.asarray(x)
            out_shape = out_shape_from_array(x)
            scalar_out = False
            # For 1d, squeeze the array
            if ndim == 1 and x.ndim == 2:
                x = x.squeeze()
        elif x in self.domain:
            x = np.atleast_2d(x).T  # make a (d, 1) array
            out_shape = (1,)
            scalar_out = (out is None)
        else:
            # Unknown input
            txt_1d = ' or (n,)' if ndim == 1 else ''
            raise TypeError('Argument {!r} not a valid vectorized '
                            'input. Expected an element of the domain '
                            '{domain}, an array-like with shape '
                            '({domain.ndim}, n){} or a length-{domain.ndim} '
                            'meshgrid tuple.'
                            ''.format(x, txt_1d, domain=self.domain))

        # Check bounds if specified
        if bounds_check:
            if not self.domain.contains_all(x):
                raise ValueError('input contains points outside '
                                 'the domain {}'.format(self.domain))

        # Call the function and check out shape, before or after
        if out is None:
            if ndim == 1:
                try:
                    out = self._call(x, **kwargs)
                except (TypeError, IndexError):
                    # TypeError is raised if a meshgrid was used but the
                    # function expected an array (1d only). In this case we try
                    # again with the first meshgrid vector.
                    # IndexError is raised in expressions like x[x > 0] since
                    # "x > 0" evaluates to 'True', i.e. 1, and that index is
                    # out of range for a meshgrid tuple of length 1 :-). To get
                    # the real errors with indexing, we check again for the
                    # same scenario (scalar output when not valid) as in the
                    # first case.
                    out = self._call(x[0], **kwargs)

                # squeeze to remove extra axes.
                out = np.squeeze(out)
            else:
                out = self._call(x, **kwargs)

            # Cast to proper dtype if needed, also convert to array if out
            # is scalar.
            out = np.asarray(out, self.out_dtype)

            if out_shape != (1,) and out.shape != out_shape:
                # Try to broadcast the returned element if possible
                out = _broadcast_to(out, out_shape)
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError('output {!r} not a `numpy.ndarray` '
                                'instance')
            if out_shape != (1,) and out.shape != out_shape:
                raise ValueError('output shape {} not equal to shape '
                                 '{} expected from input'
                                 ''.format(out.shape, out_shape))
            if self.out_dtype is not None and out.dtype != self.out_dtype:
                raise ValueError('`out.dtype` ({}) does not match out_dtype '
                                 '({})'.format(out.dtype, self.out_dtype))

            if ndim == 1:
                # TypeError for meshgrid in 1d, but expected array (see above)
                try:
                    self._call(x, out=out, **kwargs)
                except TypeError:
                    self._call(x[0], out=out, **kwargs)
            else:
                self._call(x, out=out, **kwargs)

        # Check output values
        if bounds_check and not self.space.field.contains_all(out):
                raise ValueError('output contains points outside '
                                 'the range {}'
                                 ''.format(self.field))

        # Numpy does not implement __complex__ for arrays (in contrast to
        # __float__), so we have to fish out the scalar ourselves.
        if scalar_out:
            out = out.ravel()[0]

        if scalar_out and self.space.field is not None:
            return self.space.field.element(out)
        else:
            return out

    def assign(self, other):
        """Assign ``other`` to ``self``.

        This is implemented without `FunctionSpace.lincomb` to ensure that
        ``self == other`` evaluates to True after ``self.assign(other)``.
        """
        if other not in self.space:
            raise TypeError('`other` {!r} is not an element of the space '
                            '{} of this function'
                            ''.format(other, self.space))
        self._call_in_place = other._call_in_place
        self._call_out_of_place = other._call_out_of_place
        self._call_has_out = other._call_has_out
        self._call_out_optional = other._call_out_optional

    def copy(self):
        """Create an identical (deep) copy of this element."""
        result = self.space.element()
        result.assign(self)
        return result

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `FunctionSpaceElement` with
            ``other.space == self.space``, and the functions for evaluation
            evaluation of ``self`` and ``other`` are the same, ``False``
            otherwise.
        """
        if other is self:
            return True

        if not isinstance(other, FunctionSpaceElement):
            return False

        # We cannot blindly compare since functions may have been wrapped
        if (self._call_has_out != other._call_has_out or
                self._call_out_optional != other._call_out_optional):
            return False

        if self._call_has_out:
            # Out-of-place can be wrapped in this case, so we compare only
            # the in-place methods.
            funcs_equal = self._call_in_place == other._call_in_place
        else:
            # Just the opposite of the first case
            funcs_equal = self._call_out_of_place == other._call_out_of_place

        return self.space == other.space and funcs_equal

    # Power functions are more general than the ones in LinearSpace
    def __pow__(self, p):
        """`f.__pow__(p) <==> f ** p`."""
        out = self.space.element()
        self.space._scalar_power(self, p, out=out)
        return out

    def __ipow__(self, p):
        """`f.__ipow__(p) <==> f **= p`."""
        return self.space._scalar_power(self, p, out=self)

    @property
    def real(self):
        """Pointwise real part of this function."""
        return self.space._realpart(self)

    @property
    def imag(self):
        """Pointwise imaginary part of this function."""
        return self.space._imagpart(self)

    def conj(self):
        """Pointwise complex conjugate of this function."""
        return self.space._conj(self)

    def __str__(self):
        """Return ``str(self)``."""
        if self._call_has_out:
            func = self._call_in_place
        else:
            func = self._call_out_of_place

        # Try to get a pretty-print name of the function
        fname = getattr(func, '__name__', getattr(func, 'name', str(func)))

        return '{}: {} --> {}'.format(fname, self.domain, self.out_dtype)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self._call_has_out:
            func = self._call_in_place
        else:
            func = self._call_out_of_place

        return '{!r}.element({!r})'.format(self.space, func)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
