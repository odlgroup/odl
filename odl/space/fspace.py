# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Spaces of scalar-, vector- and tensor-valued functions on a given domain."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import inspect
import numpy as np
import sys

from odl.set import RealNumbers, ComplexNumbers, Set, LinearSpace
from odl.set.space import LinearSpaceElement
from odl.util import (
    is_real_dtype, is_complex_floating_dtype, dtype_repr, dtype_str,
    complex_dtype, real_dtype, signature_string,
    is_valid_input_array, is_valid_input_meshgrid,
    out_shape_from_array, out_shape_from_meshgrid, vectorize, broadcast_to,
    writable_array)
from odl.util.utility import preload_first_arg


__all__ = ('FunctionSpace',)


def _check_out_arg(func):
    """Check if of ``func`` has an (optional) ``out`` argument.

    Also verify that the signature of ``func`` has no ``*args`` since
    they make argument propagation a hassle.

    Parameters
    ----------
    func : callable
        Object that should be inspected.

    Returns
    -------
    has_out : bool
        ``True`` if the signature has an ``out`` argument, ``False``
        otherwise.
    out_is_optional : bool
        ``True`` if ``out`` is present and optional in the signature,
        ``False`` otherwise.

    Raises
    ------
    TypeError
        If ``func``'s signature has ``*args``.
    """
    if sys.version_info.major > 2:
        spec = inspect.getfullargspec(func)
        kw_only = spec.kwonlyargs
    else:
        spec = inspect.getargspec(func)
        kw_only = ()

    if spec.varargs is not None:
        raise TypeError('*args not allowed in function signature')

    pos_args = spec.args
    pos_defaults = () if spec.defaults is None else spec.defaults

    has_out = 'out' in pos_args or 'out' in kw_only
    if 'out' in pos_args:
        has_out = True
        out_is_optional = (
            pos_args.index('out') >= len(pos_args) - len(pos_defaults))
    elif 'out' in kw_only:
        has_out = out_is_optional = True
    else:
        has_out = out_is_optional = False

    return has_out, out_is_optional


def _default_in_place(func, x, out, **kwargs):
    """Default in-place evaluation method."""
    result = np.array(func._call_out_of_place(x, **kwargs), copy=False)
    if result.dtype == object:
        # Different shapes encountered, need to broadcast
        flat_results = result.ravel()
        if is_valid_input_array(x, func.domain.ndim):
            scalar_out_shape = out_shape_from_array(x)
        elif is_valid_input_meshgrid(x, func.domain.ndim):
            scalar_out_shape = out_shape_from_meshgrid(x)
        else:
            raise RuntimeError('bad input')

        bcast_results = [broadcast_to(res, scalar_out_shape)
                         for res in flat_results]
        result = np.array(bcast_results, dtype=func.scalar_out_dtype)
        result = result.reshape(func.out_shape + scalar_out_shape)

    try:
        reshaped = result.reshape(out.shape)
    except ValueError:
        out[:] = result
    else:
        out[:] = reshaped

    return out


def _default_out_of_place(func, x, **kwargs):
    """Default in-place evaluation method."""
    if is_valid_input_array(x, func.domain.ndim):
        scalar_out_shape = out_shape_from_array(x)
    elif is_valid_input_meshgrid(x, func.domain.ndim):
        scalar_out_shape = out_shape_from_meshgrid(x)
    else:
        raise TypeError('cannot use in-place method to implement '
                        'out-of-place non-vectorized evaluation')

    dtype = func.space.scalar_out_dtype
    if dtype is None:
        dtype = np.result_type(*x)

    out_shape = func.out_shape + scalar_out_shape
    out = np.empty(out_shape, dtype=dtype)
    func._call_in_place(x, out=out, **kwargs)
    return out


def _fcall_out_type(fcall):
    """Check if ``fcall`` has (optional) output argument.

    This function is intended to work with all types of callables
    that are used as input to `FunctionSpace.element`.
    """
    if isinstance(fcall, FunctionSpaceElement):
        call_has_out = fcall._call_has_out
        call_out_optional = fcall._call_out_optional

    # Numpy Ufuncs and similar objects (e.g. Numba DUfuncs)
    elif hasattr(fcall, 'nin') and hasattr(fcall, 'nout'):
        if fcall.nin != 1:
            raise ValueError('ufunc {} has {} input parameter(s), '
                             'expected 1'
                             ''.format(fcall.__name__, fcall.nin))
        if fcall.nout > 1:
            raise ValueError('ufunc {} has {} output parameter(s), '
                             'expected at most 1'
                             ''.format(fcall.__name__, fcall.nout))
        call_has_out = call_out_optional = (fcall.nout == 1)
    elif inspect.isfunction(fcall):
        call_has_out, call_out_optional = _check_out_arg(fcall)
    elif callable(fcall):
        call_has_out, call_out_optional = _check_out_arg(fcall.__call__)
    else:
        raise TypeError('object {!r} not callable'.format(fcall))

    return call_has_out, call_out_optional


class FunctionSpace(LinearSpace):

    """A vector space of functions.

    Elements in this space represent scalar-, vector- or tensor-valued
    functions on some set, usually a subset of a Euclidean space
    :math:`\mathbb{R}^d`. The functions support vectorized evaluation,
    see `the vectorization guide
    <https://odlgroup.github.io/odl/guide/vectorization_guide.html>`_
    for details.
    """

    def __init__(self, domain, out_dtype=float):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `Set`
            The domain of the functions.
        out_dtype : optional
            Data type of the return value of a function in this
            space. Can be provided in any way the `numpy.dtype`
            constructor understands, e.g. as built-in type or as a string.

            To create a space of vector- or tensor-valued functions,
            use a dtype with a shape, e.g.,
            ``np.dtype(('float64', (2, 3)))``.

            For ``None``, the data type of function outputs is inferred
            lazily at runtime.

        Examples
        --------
        Real-valued functions on the interval [0, 1]:

        >>> domain = odl.IntervalProd(0, 1)
        >>> odl.FunctionSpace(domain)
        FunctionSpace(IntervalProd(0.0, 1.0))

        Complex-valued functions on the same domain can be created by
        specifying ``out_dtype``:

        >>> odl.FunctionSpace(domain, out_dtype=complex)
        FunctionSpace(IntervalProd(0.0, 1.0), out_dtype=complex)

        To get vector- or tensor-valued functions, specify
        ``out_dtype`` with shape:

        >>> vec_dtype = np.dtype(('float64', (3,)))  # 3 components
        >>> odl.FunctionSpace(domain, out_dtype=vec_dtype)
        FunctionSpace(IntervalProd(0.0, 1.0), out_dtype=('float64', (3,)))
        """
        if not isinstance(domain, Set):
            raise TypeError('`domain` must be a `Set` instance, got {!r}'
                            ''.format(domain))
        self.__domain = domain

        # Prevent None from being converted to float64 by np.dtype
        if out_dtype is None:
            self.__out_dtype = None
        else:
            self.__out_dtype = np.dtype(out_dtype)

        if is_real_dtype(self.out_dtype):
            field = RealNumbers()
        elif is_complex_floating_dtype(self.out_dtype):
            field = ComplexNumbers()
        else:
            field = None

        super().__init__(field)

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
        """Set from which a function in this space can take inputs."""
        return self.__domain

    @property
    def out_dtype(self):
        """Output data type of a function in this space.

        If ``None``, the output data type is not pre-defined and instead
        inferred at run-time.
        """
        return self.__out_dtype

    @property
    def scalar_out_dtype(self):
        """Scalar variant of ``out_dtype`` in case it has a shape."""
        return getattr(self.out_dtype, 'base', None)

    @property
    def real_out_dtype(self):
        """The real dtype corresponding to this space's `out_dtype`."""
        return self.__real_out_dtype

    @property
    def complex_out_dtype(self):
        """The complex dtype corresponding to this space's `out_dtype`."""
        return self.__complex_out_dtype

    @property
    def out_shape(self):
        """Shape of function values, ``()`` for scalar output."""
        return getattr(self.out_dtype, 'shape', ())

    @property
    def tensor_valued(self):
        """``True`` if functions have multi-dim. output, else ``False``."""
        return bool(self.out_shape)

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
            Default: `zero`.
        vectorized : bool, optional
            If ``True``, assume that ``fcall`` supports vectorized
            evaluation. For ``False``, , the function is decorated with a
            vectorizer, which implies that two elements created this way
            from the same function are regarded as *not equal*.
            The ``False`` option cannot be used if ``fcall`` has an
            ``out`` parameter.

        Returns
        -------
        element : `FunctionSpaceElement`
            The new element, always supports vectorization.

        Examples
        --------
        Scalar-valued functions are straightforward to create:

        >>> fspace = odl.FunctionSpace(odl.IntervalProd(0, 1))
        >>> func = fspace.element(lambda x: x - 1)
        >>> func(0.5)
        -0.5
        >>> func([0.1, 0.6])
        array([-0.9, -0.4])

        It is also possible to use functions with parameters. Note that
        such extra parameters have to be given by keyword when calling
        the function:

        >>> def f(x, b):
        ...     return x + b
        >>> func = fspace.element(f)
        >>> func([0.1, 0.6], b=1)
        array([ 1.1,  1.6])
        >>> func([0.1, 0.6], b=-1)
        array([-0.9, -0.4])

        Vector-valued functions can eiter be given as a sequence of
        scalar-valued functions or as a single function that returns
        a sequence:

        >>> # Space of vector-valued functions with 2 components
        >>> fspace = odl.FunctionSpace(odl.IntervalProd(0, 1),
        ...                            out_dtype=(float, (2,)))
        >>> # Possibility 1: provide component functions
        >>> func1 = fspace.element([lambda x: x + 1, np.negative])
        >>> func1(0.5)
        array([ 1.5, -0.5])
        >>> func1([0.1, 0.6])
        array([[ 1.1,  1.6],
               [-0.1, -0.6]])
        >>> # Possibility 2: single function returning a sequence
        >>> func2 = fspace.element(lambda x: (x + 1, -x))
        >>> func2(0.5)
        array([ 1.5, -0.5])
        >>> func2([0.1, 0.6])
        array([[ 1.1,  1.6],
               [-0.1, -0.6]])

        If the function(s) include(s) an ``out`` parameter, it can be
        provided to hold the final result:

        >>> # Sequence of functions with `out` parameter
        >>> def f1(x, out):
        ...     out[:] = x + 1
        >>> def f2(x, out):
        ...     out[:] = -x
        >>> func = fspace.element([f1, f2])
        >>> out = np.empty((2, 2))  # needs to match expected output shape
        >>> result = func([0.1, 0.6], out=out)
        >>> out
        array([[ 1.1,  1.6],
               [-0.1, -0.6]])
        >>> result is out
        True
        >>> # Single function assigning to components of `out`
        >>> def f(x, out):
        ...     out[0] = x + 1
        ...     out[1] = -x
        >>> func = fspace.element(f)
        >>> out = np.empty((2, 2))  # needs to match expected output shape
        >>> result = func([0.1, 0.6], out=out)
        >>> out
        array([[ 1.1,  1.6],
               [-0.1, -0.6]])
        >>> result is out
        True
        """
        if fcall is None:
            return self.zero()
        elif fcall in self:
            return fcall
        elif callable(fcall):
            if not vectorized:
                if hasattr(fcall, 'nin') and hasattr(fcall, 'nout'):
                    raise TypeError('`fcall` {!r} is a ufunc-like object, '
                                    'use vectorized=True')
                has_out, _ = _check_out_arg(fcall)
                if has_out:
                    raise TypeError('non-vectorized `fcall` with `out` '
                                    'parameter not allowed')
                if self.field == RealNumbers():
                    otypes = ['float64']
                elif self.field == ComplexNumbers():
                    otypes = ['complex128']
                else:
                    otypes = []

                fcall = vectorize(otypes=otypes)(fcall)
            return self.element_type(self, fcall)
        else:
            # This is for the case that an array-like of callables
            # is provided
            if np.shape(fcall) != self.out_shape:
                raise ValueError(
                    'invalid `fcall` {!r}: expected `None`, a callable or '
                    'an array-like of callables whose shape matches '
                    '`out_shape` {}'.format(self.out_shape))

            fcalls = np.array(fcall, dtype=object, ndmin=1).ravel().tolist()
            if not vectorized:
                if self.field == RealNumbers():
                    otypes = ['float64']
                elif self.field == ComplexNumbers():
                    otypes = ['complex128']
                else:
                    otypes = []

                # Vectorize, preserving scalars
                fcalls = [f if np.isscalar(f) else vectorize(otypes=otypes)(f)
                          for f in fcalls]

            if sys.version_info.major < 3:
                getargspec = inspect.getargspec
            else:
                getargspec = inspect.getfullargspec

            def wrapper(x, out=None, **kwargs):
                """Function wrapping an array of callables."""
                if is_valid_input_meshgrid(x, self.domain.ndim):
                    scalar_out_shape = out_shape_from_meshgrid(x)
                elif is_valid_input_array(x, self.domain.ndim):
                    scalar_out_shape = out_shape_from_array(x)
                else:
                    raise RuntimeError('bad input')

                if out is None:
                    # Out-of-place evaluation

                    # Collect results of member functions into a list
                    results = []
                    for f in fcalls:
                        if np.isscalar(f):
                            # Constant function
                            results.append(f)
                        elif not callable(f):
                            raise TypeError('element {!r} of sequence not '
                                            'callable'.format(f))
                        elif hasattr(f, 'nin') and hasattr(f, 'nout'):
                            # ufunc-like object
                            results.append(f(x, **kwargs))
                        else:
                            try:
                                has_out = 'out' in getargspec(f).args
                            except TypeError:
                                raise TypeError('unsupported callable {!r}'
                                                ''.format(f))
                            else:
                                if has_out:
                                    out = np.empty(scalar_out_shape,
                                                   dtype=self.scalar_out_dtype)
                                    f(x, out=out, **kwargs)
                                    results.append(out)
                                else:
                                    results.append(f(x, **kwargs))

                    # Broadcast to required shape and convert to array.
                    # This will raise an error if the shape of some member
                    # array is wrong, since in that case the resulting
                    # dtype would be `object`.
                    bcast_results = []
                    for res in results:
                        try:
                            reshaped = np.reshape(res, scalar_out_shape)
                        except ValueError:
                            bcast_results.append(
                                broadcast_to(res, scalar_out_shape))
                        else:
                            bcast_results.append(reshaped)

                    out_arr = np.array(bcast_results,
                                       dtype=self.scalar_out_dtype)

                    return out_arr.reshape(self.out_shape + scalar_out_shape)

                else:
                    # In-place evaluation

                    # This is a precaution in case out is not contiguous
                    with writable_array(out) as out_arr:
                        # Flatten tensor axes to work on one tensor
                        # component (= scalar function) at a time
                        out_comps = out.reshape((-1,) + scalar_out_shape)
                        for f, out_comp in zip(fcalls, out_comps):
                            if np.isscalar(f):
                                out_comp[:] = f
                            else:
                                has_out, _ = _fcall_out_type(f)
                                if has_out:
                                    f(x, out=out_comp, **kwargs)
                                else:
                                    out_comp[:] = f(x, **kwargs)

            return self.element_type(self, wrapper)

    def zero(self):
        """Function mapping anything to zero."""
        # Since `FunctionSpace.lincomb` may be slow, we implement this
        # function directly.
        # The unused **kwargs are needed to support combination with
        # functions that take parameters.
        def zero_vec(x, out=None, **kwargs):
            """Zero function, vectorized."""
            if is_valid_input_meshgrid(x, self.domain.ndim):
                scalar_out_shape = out_shape_from_meshgrid(x)
            elif is_valid_input_array(x, self.domain.ndim):
                scalar_out_shape = out_shape_from_array(x)
            else:
                raise TypeError('invalid input type')

            # For tensor-valued functions
            out_shape = self.out_shape + scalar_out_shape

            if out is None:
                return np.zeros(out_shape, dtype=self.scalar_out_dtype)
            else:
                # Need to go through an array to fill with the correct
                # zero value for all dtypes
                fill_value = np.zeros(1, dtype=self.scalar_out_dtype)[0]
                out.fill(fill_value)

        return self.element_type(self, zero_vec)

    def one(self):
        """Function mapping anything to one."""
        # See zero() for remarks
        def one_vec(x, out=None, **kwargs):
            """One function, vectorized."""
            if is_valid_input_meshgrid(x, self.domain.ndim):
                scalar_out_shape = out_shape_from_meshgrid(x)
            elif is_valid_input_array(x, self.domain.ndim):
                scalar_out_shape = out_shape_from_array(x)
            else:
                raise TypeError('invalid input type')

            # For tensor-valued functions
            out_shape = self.out_shape + scalar_out_shape

            if out is None:
                return np.ones(out_shape, dtype=self.scalar_out_dtype)
            else:
                # Need to go through an array to fill with the correct
                # zero value for all dtypes
                fill_value = np.ones(1, dtype=self.scalar_out_dtype)[0]
                out.fill(fill_value)

        return self.element_type(self, one_vec)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `FunctionSpace` with same
            `FunctionSpace.domain`, `FunctionSpace.field` and
            `FunctionSpace.out_dtype`, ``False`` otherwise.
        """
        if other is self:
            return True

        return (type(other) == type(self) and
                self.domain == other.domain and
                self.field == other.field and
                self.out_dtype == other.out_dtype)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.domain, self.field, self.out_dtype))

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
            way `numpy.dtype` understands, e.g. as string (``'complex64'``)
            or built-in type (``complex``). ``None`` is interpreted as
            ``'float64'``.

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

    def _lincomb(self, a, f1, b, f2, out):
        """Linear combination of ``f1`` and ``f2``.

        Notes
        -----
        The additions and multiplications are implemented via simple
        Python functions, so non-vectorized versions are slow.
        """
        # Avoid infinite recursions by making a copy of the functions
        f1_copy = f1.copy()
        f2_copy = f2.copy()

        def lincomb_oop(x, **kwargs):
            """Linear combination, out-of-place version."""
            # Not optimized since that raises issues with alignment
            # of input and partial results
            out = a * np.asarray(f1_copy(x, **kwargs),
                                 dtype=self.scalar_out_dtype)
            tmp = b * np.asarray(f2_copy(x, **kwargs),
                                 dtype=self.scalar_out_dtype)
            out += tmp
            return out

        out._call_out_of_place = lincomb_oop
        decorator = preload_first_arg(out, 'in-place')
        out._call_in_place = decorator(_default_in_place)
        out._call_has_out = out._call_out_optional = False
        return out

    def _multiply(self, f1, f2, out):
        """Pointwise multiplication of ``f1`` and ``f2``.

        Notes
        -----
        The multiplication is implemented with a simple Python
        function, so the non-vectorized versions are slow.
        """
        # Avoid infinite recursions by making a copy of the functions
        f1_copy = f1.copy()
        f2_copy = f2.copy()

        def product_oop(x, **kwargs):
            """Product out-of-place evaluation function."""
            return np.asarray(f1_copy(x, **kwargs) * f2_copy(x, **kwargs),
                              dtype=self.scalar_out_dtype)

        out._call_out_of_place = product_oop
        decorator = preload_first_arg(out, 'in-place')
        out._call_in_place = decorator(_default_in_place)
        out._call_has_out = out._call_out_optional = False
        return out

    def _divide(self, f1, f2, out):
        """Pointwise division of ``f1`` and ``f2``.

        Notes
        -----
        The division is implemented with a simple Python
        function, so the non-vectorized versions are slow.
        """
        # Avoid infinite recursions by making a copy of the functions
        f1_copy = f1.copy()
        f2_copy = f2.copy()

        def quotient_oop(x, **kwargs):
            """Quotient out-of-place evaluation function."""
            return np.asarray(f1_copy(x, **kwargs) / f2_copy(x, **kwargs),
                              dtype=self.scalar_out_dtype)

        out._call_out_of_place = quotient_oop
        decorator = preload_first_arg(out, 'in-place')
        out._call_in_place = decorator(_default_in_place)
        out._call_has_out = out._call_out_optional = False
        return out

    def _scalar_power(self, f, p, out):
        """Compute ``p``-th power of ``f`` for ``p`` scalar."""
        # Avoid infinite recursions by making a copy of the function
        f_copy = f.copy()

        def pow_posint(x, n):
            """Power function for positive integer ``n``, out-of-place."""
            if isinstance(x, np.ndarray):
                y = x.copy()
                return ipow_posint(y, n)
            else:
                return x ** n

        def ipow_posint(x, n):
            """Power function for positive integer ``n``, in-place."""
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

        def power_oop(x, **kwargs):
            """Power out-of-place evaluation function."""
            if p == 0:
                return self.one()
            elif p == int(p) and p >= 1:
                return np.asarray(pow_posint(f_copy(x, **kwargs), int(p)),
                                  dtype=self.scalar_out_dtype)
            else:
                result = np.power(f_copy(x, **kwargs), p)
                return result.astype(self.scalar_out_dtype)

        out._call_out_of_place = power_oop
        decorator = preload_first_arg(out, 'in-place')
        out._call_in_place = decorator(_default_in_place)
        out._call_has_out = out._call_out_optional = False
        return out

    def _realpart(self, f):
        """Function returning the real part of the result from ``f``."""
        # Avoid infinite recursions by making a copy of the function
        f_copy = f.copy()

        def f_re(x, **kwargs):
            result = np.asarray(f_copy(x, **kwargs),
                                dtype=self.scalar_out_dtype)
            return result.real

        if is_real_dtype(self.out_dtype):
            return f
        else:
            return self.real_space.element(f_re)

    def _imagpart(self, f):
        """Function returning the imaginary part of the result from ``f``."""
        # Avoid infinite recursions by making a copy of the function
        f_copy = f.copy()

        def f_im(x, **kwargs):
            result = np.asarray(f_copy(x, **kwargs),
                                dtype=self.scalar_out_dtype)
            return result.imag

        if is_real_dtype(self.out_dtype):
            return self.zero()
        else:
            return self.real_space.element(f_im)

    def _conj(self, f):
        """Function returning the complex conjugate of a result."""
        # Avoid infinite recursions by making a copy of the function
        f_copy = f.copy()

        def f_conj(x, **kwargs):
            result = np.asarray(f_copy(x, **kwargs),
                                dtype=self.scalar_out_dtype)
            return result.conj()

        if is_real_dtype(self.out_dtype):
            return f
        else:
            return self.element(f_conj)

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
        # TODO: adapt for tensor-valued functions

        # Get the points and calculate some statistics on them
        mins = self.domain.min()
        maxs = self.domain.max()
        means = (maxs + mins) / 2.0
        stds = (maxs - mins) / 4.0
        ndim = getattr(self.domain, 'ndim', 0)

        # Zero and One
        yield ('Zero', self.zero())
        yield ('One', self.one())

        # Indicator function in first dimension
        def step_fun(x):
            return (x[0] > means[0])

        yield ('Step', self.element(step_fun))

        # Indicator function on hypercube
        def cube_fun(x):
            result = True
            for points, mean, std in zip(x, means, stds):
                result = np.logical_and(result, points < mean + std)
                result = np.logical_and(result, points > mean - std)
            return result

        yield ('Cube', self.element(cube_fun))

        # Indicator function on a ball
        if ndim > 1:  # Only if ndim > 1, don't duplicate cube
            def ball_fun(x):
                r = sum((xi - mean) ** 2 / std ** 2
                        for xi, mean, std in zip(x, means, stds))
                return r < 1.0

            yield ('Ball', self.element(ball_fun))

        # Gaussian function
        def gaussian_fun(x):
            r2 = sum((xi - mean) ** 2 / (2 * std ** 2)
                     for xi, mean, std in zip(x, means, stds))
            return np.exp(-r2)

        yield ('Gaussian', self.element(gaussian_fun))

        # Gradient in each dimensions
        for axis in range(ndim):
            def gradient_fun(x):
                return (x[axis] - mins[axis]) / (maxs[axis] - mins[axis])

            yield ('Grad {}'.format(axis), self.element(gradient_fun))

        # Gradient in all dimensions
        if ndim > 1:  # Only if ndim > 1, don't duplicate grad 0
            def all_gradient_fun(x):
                return sum((xi - xmin) / (xmax - xmin)
                           for xi, xmin, xmax in zip(x, mins, maxs))

            yield ('Grad all', self.element(all_gradient_fun))

    @property
    def element_type(self):
        """`FunctionSpaceElement`"""
        return FunctionSpaceElement

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.domain]
        optargs = []
        if is_real_dtype(self.out_dtype):
            default_field = RealNumbers()
            default_dtype_string = 'float'
        elif is_complex_floating_dtype(self.out_dtype):
            default_field = ComplexNumbers()
            default_dtype_string = 'None'
        else:
            default_field = None
            default_dtype_string = 'None'

        optargs.append(('field', self.field, default_field))
        dtype_string = dtype_str(self.out_dtype)
        optargs.append(('out_dtype', dtype_string, default_dtype_string))

        inner_str = signature_string(posargs, optargs,
                                     mod=[['!r'], ['', '!s']])
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class FunctionSpaceElement(LinearSpaceElement):

    """Representation of a `FunctionSpace` element."""

    def __init__(self, fspace, fcall):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            Set of functions this element lives in.
        fcall : callable
            Object used to evaluate the function. Must support
            vectorization and accept a sequence of
            coordinate arrays ``x[0], ..., x[d]`` in sparse or dense
            form, and return (or write to the ``out`` array) an
            array of appropriate shape.
        """
        super().__init__(fspace)
        self._call_has_out, self._call_out_optional = _fcall_out_type(fcall)

        if not self._call_has_out:
            # Out-of-place-only
            decorator = preload_first_arg(self, 'in-place')
            self._call_in_place = decorator(_default_in_place)
            self._call_out_of_place = fcall
        elif self._call_out_optional:
            # Dual-use
            self._call_in_place = self._call_out_of_place = fcall
        else:
            # In-place-only
            self._call_in_place = fcall
            decorator = preload_first_arg(self, 'out-of-place')
            self._call_out_of_place = decorator(_default_out_of_place)

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

    @property
    def scalar_out_dtype(self):
        """Scalar variant of ``out_dtype`` in case it has a shape."""
        return self.space.scalar_out_dtype

    @property
    def out_shape(self):
        """Shape of function values, ``()`` for scalar output."""
        return self.space.out_shape

    @property
    def tensor_valued(self):
        """``True`` if the output is multi-dim. output, else ``False``."""
        return self.space.tensor_valued

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
        bounds_check : bool
            If ``True``, check if all input points lie in the function
            domain in the case of vectorized evaluation. This requires
            the domain to implement `Set.contains_all`.
            Default: ``True``

        Returns
        -------
        out : `range` element or `numpy.ndarray` of elements
            Result of the function evaluation. If ``out`` was provided,
            the returned object is a reference to it.

        Raises
        ------
        TypeError
            If ``x`` is not a valid vectorized evaluation argument.

            If ``out`` is neither ``None`` nor a `numpy.ndarray` of
            adequate shape and data type.

        ValueError
            If ``bounds_check == True`` evaluation points fall outside
            the valid domain.
        """
        bounds_check = kwargs.pop('bounds_check', self.space.field is not None)
        if bounds_check and not hasattr(self.domain, 'contains_all'):
            raise AttributeError('bounds check not possible for '
                                 'domain {}, missing `contains_all()` '
                                 'method'.format(self.domain))

        if bounds_check and not hasattr(self.space.field, 'contains_all'):
            raise AttributeError('bounds check not possible for '
                                 'field {}, missing `contains_all()` '
                                 'method'.format(self.space.field))

        ndim = getattr(self.domain, 'ndim', None)
        # Check for input type and determine output shape
        if is_valid_input_meshgrid(x, ndim):
            scalar_in = False
            scalar_out_shape = out_shape_from_meshgrid(x)
            scalar_out = False
            # Avoid operations on tuples like x * 2 by casting to array
            if ndim == 1:
                x = x[0][None, ...]
        elif is_valid_input_array(x, ndim):
            x = np.asarray(x)
            scalar_in = False
            scalar_out_shape = out_shape_from_array(x)
            scalar_out = False
        elif x in self.domain:
            x = np.atleast_2d(x).T  # make a (d, 1) array
            scalar_in = True
            scalar_out_shape = (1,)
            scalar_out = (out is None and not self.space.tensor_valued)
        else:
            # Unknown input
            txt_1d = ' or (n,)' if ndim == 1 else ''
            raise TypeError('argument {!r} not a valid function '
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

        if scalar_in:
            out_shape = self.out_shape
        else:
            out_shape = self.out_shape + scalar_out_shape

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

            else:
                # Here we don't catch exceptions since they are likely true
                # errors
                out = self._call(x, **kwargs)

            if isinstance(out, np.ndarray) or np.isscalar(out):
                # Cast to proper dtype if needed, also convert to array if out
                # is a scalar.
                out = np.asarray(out, dtype=self.space.scalar_out_dtype)
                if scalar_in:
                    out = np.squeeze(out)
                elif ndim == 1 and out.shape == (1,) + out_shape:
                    out = out.reshape(out_shape)

                if out_shape not in ((), (1,)) and out.shape != out_shape:
                    # Try to broadcast the returned element.
                    out = broadcast_to(out, out_shape)

            elif self.space.tensor_valued:
                # TODO: fix case when such a function is evaluated at a
                # single point

                # The out object can be any array-like of objects with shapes
                # that should all be broadcastable to scalar_out_shape.
                results = np.array(out)
                if results.dtype == object or scalar_in:
                    # Some results don't have correct shape, need to
                    # broadcast
                    bcast_res = []
                    for res in results.ravel():
                        if ndim == 1:
                            # As usual, 1d is tedious to deal with. This
                            # code deals with extra dimensions in result
                            # components that stem from using x instead of
                            # x[0] in a function.
                            # Without this, broadcasting fails.
                            shp = getattr(res, 'shape', ())
                            if shp and shp[0] == 1:
                                res = res.reshape(res.shape[1:])
                        bcast_res.append(broadcast_to(res, scalar_out_shape))

                    out_arr = np.array(bcast_res,
                                       dtype=self.space.scalar_out_dtype)
                elif (self.scalar_out_dtype is not None and
                      results.dtype != self.scalar_out_dtype):
                    raise ValueError(
                        'result is of dtype {}, expected {}'
                        ''.format(dtype_repr(results.dtype),
                                  dtype_repr(self.space.scalar_out_dtype)))
                else:
                    out_arr = results

                out = out_arr.reshape(out_shape)

            else:
                # TODO: improve message
                raise RuntimeError('bad output of function call')

        else:
            if not isinstance(out, np.ndarray):
                raise TypeError('output {!r} not a `numpy.ndarray` '
                                'instance')
            if out_shape != (1,) and out.shape != out_shape:
                raise ValueError('output shape {} not equal to shape '
                                 '{} expected from input'
                                 ''.format(out.shape, out_shape))
            if (self.out_dtype is not None and
                    out.dtype != self.scalar_out_dtype):
                raise ValueError('`out.dtype` ({}) does not match out_dtype '
                                 '({})'.format(out.dtype, self.out_dtype))

            if ndim == 1 and not self.tensor_valued:
                # TypeError for meshgrid in 1d, but expected array (see above)
                try:
                    self._call(x, out=out, **kwargs)
                except TypeError:
                    self._call(x[0], out=out, **kwargs)
            else:
                self._call(x, out=out, **kwargs)

        # Check output values
        if bounds_check:
            if not self.space.field.contains_all(out):
                raise ValueError('output contains values not in the field '
                                 '{}'
                                 ''.format(self.space.field))

        # Numpy < 1.12 does not implement __complex__ for arrays (in contrast
        # to __float__), so we have to fish out the scalar ourselves.
        if scalar_out:
            if self.space.field is None:
                return out.ravel()[0]
            else:
                return self.space.field.element(out.ravel()[0])
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
            of ``self`` and ``other`` are the same, ``False``
            otherwise.

        Notes
        -----
        Since there is potentially a lot of function wrapping going on,
        it is very hard to find the "true" function behind a
        `FunctionSpaceElement` for comparison. Therefore, users
        should be aware that very often, comparison evaluates to ``False``
        even if two elements were generated from the same function.
        """
        if other is self:
            return True
        elif other not in self.space:
            return False

        # We try to unwrap one level, which is better than nothing
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
