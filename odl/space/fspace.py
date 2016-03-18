﻿# Copyright 2014-2016 The ODL development group
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

"""Spaces of functions with common domain and range."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External imports
import numpy as np
from inspect import isfunction

# ODL imports
from odl.operator.operator import Operator, _dispatch_call_args
from odl.set.sets import RealNumbers, ComplexNumbers, Set, Field
from odl.set.space import LinearSpace, LinearSpaceVector
from odl.space.base_ntuples import _TYPE_MAP_R2C, _TYPE_MAP_C2R
from odl.util.utility import (is_real_dtype, is_complex_floating_dtype,
                              preload_first_arg, dtype_repr)
from odl.util.vectorization import (
    is_valid_input_array, is_valid_input_meshgrid,
    out_shape_from_array, out_shape_from_meshgrid, vectorize)


__all__ = ('FunctionSet', 'FunctionSetVector',
           'FunctionSpace', 'FunctionSpaceVector')


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
                        'out-of-place non-vectorized evaluation.')

    dtype = func.space.out_dtype
    if dtype is None:
        dtype = np.result_type(*x)

    out = np.empty(out_shape, dtype=dtype)
    func(x, out=out, **kwargs)
    return out


class FunctionSet(Set):

    """A general set of functions with common domain and range."""

    def __init__(self, domain, range, out_dtype=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `Set`
            The domain of the functions.
        range : `Set`
            The range of the functions.
        out_dtype : optional
            Data type of the return value of a function in this space.
            Can be given in any way `np.dtype` understands, e.g. as
            string ('bool') or data type (`bool`).
            If no data type is given, a "lazy" evaluation is applied,
            i.e. an adequate data type is inferred during function
            evaluation.
        """
        if not isinstance(domain, Set):
            raise TypeError('domain {!r} not a `Set` instance.'.format(domain))

        if not isinstance(range, Set):
            raise TypeError('range {!r} not a `Set` instance.'.format(range))

        self._domain = domain
        self._range = range
        self._out_dtype = None if out_dtype is None else np.dtype(out_dtype)

    @property
    def domain(self):
        """Common domain of all functions in this set."""
        return self._domain

    @property
    def range(self):
        """Common range of all functions in this set."""
        return self._range

    @property
    def out_dtype(self):
        """Output data type of functions in this space."""
        return self._out_dtype

    def element(self, fcall=None, vectorized=True):
        """Create a `FunctionSet` element.

        Parameters
        ----------
        fcall : `callable`, optional
            The actual instruction for out-of-place evaluation.
            It must return an `FunctionSet.range` element or a
            `numpy.ndarray` of such (vectorized call).

        vectorized : bool
            Whether ``fcall`` supports vectorized evaluation.

        Returns
        -------
        element : `FunctionSetVector`
            The new element, always supports vectorization

        See also
        --------
        odl.discr.grid.TensorGrid.meshgrid : efficient grids for function
            evaluation
        """
        if not callable(fcall):
            raise TypeError('function {!r} is not callable.'.format(fcall))

        if not vectorized:
            fcall = vectorize(fcall)

        return self.element_type(self, fcall)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `FunctionSet` with same
            `FunctionSet.domain` and `FunctionSet.range`, `False` otherwise.
        """
        if other is self:
            return True

        return (isinstance(other, FunctionSet) and
                self.domain == other.domain and
                self.range == other.range and
                self.out_dtype == other.out_dtype)

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `FunctionSetVector`
            whose `FunctionSetVector.space` attribute
            equals this space, `False` otherwise.
        """
        return (isinstance(other, self.element_type) and
                self == other.space)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.range)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.domain, self.range)

    @property
    def element_type(self):
        """`FunctionSetVector`"""
        return FunctionSetVector


class FunctionSetVector(Operator):

    """Representation of a `FunctionSet` element."""

    def __init__(self, fset, fcall, out_dtype=None):
        """Initialize a new instance.

        Parameters
        ----------
        fset : `FunctionSet`
            The set of functions this element lives in
        fcall : `callable`
            The actual instruction for out-of-place evaluation.
            It must return an `FunctionSet.range` element or a
            `numpy.ndarray` of such (vectorized call).
        out_d
        """
        self._space = fset
        super().__init__(self._space.domain, self._space.range, linear=False)

        # Determine which type of implementation fcall is
        if isinstance(fcall, FunctionSetVector):
            call_has_out, call_out_optional, _ = _dispatch_call_args(
                bound_call=fcall._call)

        # Numpy Ufuncs and similar objects (e.g. Numba DUfuncs)
        elif hasattr(fcall, 'nin') and hasattr(fcall, 'nout'):
            if fcall.nin != 1:
                raise ValueError('ufunc {} has {} input parameter(s), '
                                 'expected 1.'
                                 ''.format(fcall.__name__, fcall.nin))
            if fcall.nout > 1:
                raise ValueError('ufunc {} has {} output parameter(s), '
                                 'expected at most 1.'
                                 ''.format(fcall.__name__, fcall.nout))
            call_has_out = call_out_optional = (fcall.nout == 1)
        elif isfunction(fcall):
            call_has_out, call_out_optional, _ = _dispatch_call_args(
                unbound_call=fcall)
        elif callable(fcall):
            call_has_out, call_out_optional, _ = _dispatch_call_args(
                bound_call=fcall.__call__)
        else:
            raise TypeError('type {!r} not callable.')

        self._call_has_out = call_has_out
        self._call_out_optional = call_out_optional

        if not call_has_out:
            # Out-of-place only
            self._call_in_place = preload_first_arg(self, 'in-place')(
                _default_in_place)
            self._call_out_of_place = fcall
        elif call_out_optional:
            # Dual-use
            self._call_in_place = self._call_out_of_place = fcall
        else:
            # In-place only
            self._call_in_place = fcall
            # The default out-of-place method needs to guess the data
            # type, so we need a separate decorator to help it.
            self._call_out_of_place = preload_first_arg(self, 'out-of-place')(
                _default_out_of_place)

    @property
    def space(self):
        """The space or set this function belongs to."""
        return self._space

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
        x : domain `element-like`, `meshgrid` or `numpy.ndarray`
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

        bounds_check : `bool`
            If `True`, check if all input points lie in the function
            domain in the case of vectorized evaluation. This requires
            the domain to implement `Set.contains_all`.
            Default: `True`

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
        bounds_check = kwargs.pop('bounds_check', True)
        if bounds_check and not hasattr(self.domain, 'contains_all'):
            raise AttributeError('bounds check not possible for '
                                 'domain {}, missing `contains_all()` '
                                 'method.'.format(self.domain))

        ndim = getattr(self.domain, 'ndim', None)
        # Check for input type and determine output shape
        if is_valid_input_meshgrid(x, ndim):
            out_shape = out_shape_from_meshgrid(x)
            scalar_out = False
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
            raise TypeError('argument {!r} not a valid vectorized '
                            'input. Expected an element of the domain '
                            '{dom}, an array-like with shape '
                            '({dom.ndim}, n){} or a length-{dom.ndim} '
                            'meshgrid tuple.'
                            ''.format(x, txt_1d, dom=self.domain))

        # Check bounds if specified
        if bounds_check:
            if not self.domain.contains_all(x):
                raise ValueError('input contains points outside '
                                 'the domain {}.'.format(self.domain))

        # Call the function and check out shape, before or after
        if out is None:
            try:
                if ndim == 1:
                    out = self._call(x, **kwargs)
                    if np.ndim(out) == 0 and not scalar_out:
                        # Don't accept scalar result. A typical situation where
                        # this occurs is with comparison operators, e.g.
                        # "return x > 0" which simply gives 'True' for a
                        # non-empty tuple (in Python 2). We raise TypeError
                        # to trigger the call with x[0].
                        raise TypeError
                    out = np.atleast_1d(np.squeeze(out))
                else:
                    out = self._call(x, **kwargs)
            except (TypeError, IndexError) as err:
                # TypeError is raised if a meshgrid was used but the function
                # expected an array (1d only). In this case we try again with
                # the first meshgrid vector.
                # IndexError is raised in expressions like x[x > 0] since
                # "x > 0" evaluates to 'True', i.e. 1, and that index is
                # out of range for a meshgrid tuple of length 1 :-). To get
                # the real errors with indexing, we check again for the same
                # scenario (scalar output when not valid) as in the first case.
                if ndim == 1:
                    out = self._call(x[0], **kwargs)
                    if np.ndim(out) == 0 and not scalar_out:
                        raise ValueError('invalid scalar output.')
                    out = np.atleast_1d(np.squeeze(out))
                else:
                    raise err

            if out_shape != (1,) and out.shape != out_shape:
                raise ValueError('output shape {} not equal to shape '
                                 '{} expected from input.'
                                 ''.format(out.shape, out_shape))
        else:
            if not isinstance(out, np.ndarray):
                raise TypeError('output {!r} not a `numpy.ndarray` '
                                'instance.')
            if out_shape != (1,) and out.shape != out_shape:
                raise ValueError('output shape {} not equal to shape '
                                 '{} expected from input.'
                                 ''.format(out.shape, out_shape))
            try:
                self._call(x, out=out, **kwargs)
            except TypeError as err:
                # TypeError for meshgrid in 1d, but expected array (see above)
                if ndim == 1:
                    self._call(x[0], out=out, **kwargs)
                else:
                    raise err

        # Check output values
        if bounds_check:
            if not self.range.contains_all(out):
                raise ValueError('output contains points outside '
                                 'the range {}.'
                                 ''.format(self.range))

        # Numpy does not implement __complex__ for arrays (in contrast to
        # __float__), so we have to fish out the scalar ourselves.
        return self.range.element(out.ravel()[0]) if scalar_out else out

    def assign(self, other):
        """Assign ``other`` to this vector.

        This is implemented without `FunctionSpace.lincomb` to ensure that
        ``vec == other`` evaluates to `True` after
        ``vec.assign(other)``.
        """
        if other not in self.space:
            raise TypeError('vector {!r} is not an element of the space '
                            '{} of this vector.'
                            ''.format(other, self.space))
        self._call_in_place = other._call_in_place
        self._call_out_of_place = other._call_out_of_place
        self._call_has_out = other._call_has_out
        self._call_out_optional = other._call_out_optional

    def copy(self):
        """Create an identical (deep) copy of this vector."""
        result = self.space.element()
        result.assign(self)
        return result

    def __eq__(self, other):
        """Returns ``vec == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `FunctionSetVector` with
            ``other.space`` equal to this vector's space and evaluation
            function of ``other`` and this vector is equal. `False`
            otherwise.
        """
        if other is self:
            return True

        if not isinstance(other, FunctionSetVector):
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

    def __str__(self):
        """Return ``str(self)``"""
        if self._call_has_out:
            func = self._call_in_place
        else:
            func = self._call_out_of_place
        return '{}: {} --> {}'.format(func, self.domain, self.range)

    def __repr__(self):
        """Return ``repr(self)``"""
        if self._call_has_out:
            func = self._call_in_place
        else:
            func = self._call_out_of_place

        return '{!r}.element({!r})'.format(self.space, func)


class FunctionSpace(FunctionSet, LinearSpace):

    """A vector space of functions."""

    def __init__(self, domain, field=None, out_dtype=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `Set`
            The domain of the functions
        field : `Field`, optional
            The range of the functions, usually the `RealNumbers` or
            `ComplexNumbers`. If not given, the field is either inferred
            from ``out_dtype``, or, if the latter is also `None`, set
            to ``RealNumbers()``.
        out_dtype : optional
            Data type of the return value of a function in this space.
            Can be given in any way `np.dtype` understands, e.g. as
            string ('float64') or data type (`float`).
            By default, 'float64' is used for real and 'complex128'
            for complex spaces.
        """
        if not isinstance(domain, Set):
            raise TypeError('domain {!r} not a Set instance.'.format(domain))

        if field is not None and not isinstance(field, Field):
            raise TypeError('field {!r} not a `Field` instance.'
                            ''.format(field))

        # Data type: check if consistent with field, take default for None
        dtype, dtype_in = np.dtype(out_dtype), out_dtype

        # Default for both None
        if field is None and out_dtype is None:
            field = RealNumbers()
            out_dtype = np.dtype('float64')

        # field None, dtype given -> infer field
        elif field is None:
            if is_real_dtype(dtype):
                field = RealNumbers()
            elif is_complex_floating_dtype(dtype):
                field = ComplexNumbers()
            else:
                raise ValueError('{} is not a scalar data type.'
                                 ''.format(dtype_in))

        # field given -> infer dtype if not given, else check consistency
        elif field == RealNumbers():
            if out_dtype is None:
                out_dtype = np.dtype('float64')
            elif not is_real_dtype(dtype):
                raise ValueError('{} is not a real data type.'
                                 ''.format(dtype_in))
        elif field == ComplexNumbers():
            if out_dtype is None:
                out_dtype = np.dtype('complex128')
            elif not is_complex_floating_dtype(dtype):
                raise ValueError('{} is not a complex data type.'
                                 ''.format(dtype_in))

        # Else: keep out_dtype=None, which results in lazy dtype determination

        LinearSpace.__init__(self, field)
        FunctionSet.__init__(self, domain, field, out_dtype)

        # Init cache attributes for real / complex variants
        if self.field == RealNumbers():
            self._real_out_dtype = self.out_dtype
            self._real_space = self
            self._complex_out_dtype = _TYPE_MAP_R2C.get(self.out_dtype,
                                                        np.dtype(object))
            self._complex_space = None
        elif self.field == ComplexNumbers():
            self._real_out_dtype = _TYPE_MAP_C2R[self.out_dtype]
            self._real_space = None
            self._complex_out_dtype = self.out_dtype
            self._complex_space = self
        else:
            self._real_out_dtype = None
            self._real_space = None
            self._complex_out_dtype = None
            self._complex_space = None

    def element(self, fcall=None, vectorized=True):
        """Create a `FunctionSpace` element.

        Parameters
        ----------
        fcall : `callable`, optional
            The actual instruction for out-of-place evaluation.
            It must return an `FunctionSet.range` element or a
            `numpy.ndarray` of such (vectorized call).

            If fcall is a `FunctionSetVector`, it is wrapped
            as a new `FunctionSpaceVector`.

        vectorized : bool
            Whether ``fcall`` supports vectorized evaluation.

        Returns
        -------
        element : `FunctionSpaceVector`
            The new element, always supports vectorization

        Notes
        -----
        If you specify ``vectorized=False``, the function is decorated
        with a vectorizer, which makes two elements created this way
        from the same function being regarded as *not equal*.
        """
        if fcall is None:
            return self.zero()
        else:
            if not callable(fcall):
                raise TypeError('function {!r} is not callable.'.format(fcall))
            if not vectorized:
                if self.field == RealNumbers():
                    dtype = 'float64'
                else:
                    dtype = 'complex128'

                fcall = vectorize(otypes=[dtype])(fcall)

            return self.element_type(self, fcall)

    def zero(self):
        """The function mapping everything to zero.

        This function is the additive unit in the function space.

        Since `FunctionSpace.lincomb` may be slow, we implement this function
        directly.
        """
        def zero_vec(x, out=None):
            """The zero function, vectorized."""
            if is_valid_input_meshgrid(x, self.domain.ndim):
                out_shape = out_shape_from_meshgrid(x)
            elif is_valid_input_array(x, self.domain.ndim):
                out_shape = out_shape_from_array(x)
            else:
                raise TypeError('invalid input type.')

            if out is None:
                return np.zeros(out_shape, dtype=self.out_dtype)
            else:
                out.fill(0)

        return self.element_type(self, zero_vec)

    def one(self):
        """The function mapping everything to one.

        This function is the multiplicative unit in the function space.
        """
        def one_vec(x, out=None):
            """The one function, vectorized."""
            if is_valid_input_meshgrid(x, self.domain.ndim):
                out_shape = out_shape_from_meshgrid(x)
            elif is_valid_input_array(x, self.domain.ndim):
                out_shape = out_shape_from_array(x)
            else:
                raise TypeError('invalid input type.')

            if out is None:
                return np.ones(out_shape, dtype=self.out_dtype)
            else:
                out.fill(1)

        return self.element_type(self, one_vec)

    def __eq__(self, other):
        """Returns ``s == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `FunctionSpace` with same
            `FunctionSpace.domain` and `FunctionSpace.range`,
            `False` otherwise.
        """
        if other is self:
            return True

        return (isinstance(other, FunctionSpace) and
                FunctionSet.__eq__(self, other))

    def _astype(self, out_dtype):
        """Internal helper for ``astype``."""
        return type(self)(self.domain, out_dtype=out_dtype)

    def astype(self, out_dtype):
        """Return a copy of this space with new ``out_dtype``.

        Parameters
        ----------
        out_dtype : optional
            Output data type of the returned space. Can be given in any
            way `numpy.dtype` understands, e.g. as string ('complex64')
            or data type (`complex`). `None` is interpreted as 'float64'.

        Returns
        -------
        newspace : `FunctionSpace`
            The version of this space with given data type
        """
        out_dtype = np.dtype(out_dtype)
        if out_dtype == self.out_dtype:
            return self

        # Caching for real and complex versions (exact dtyoe mappings)
        if out_dtype == self._real_out_dtype:
            if self._real_space is None:
                self._real_space = self._astype(out_dtype)
            return self._real_space
        elif out_dtype == self._complex_out_dtype:
            if self._complex_space is None:
                self._complex_space = self._astype(out_dtype)
            return self._complex_space
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
                                'is of type `numpy.ndarray`.')
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
            """The product out-of-place evaluation function."""
            return np.asarray(x1_call_oop(x) * x2_call_oop(x),
                              dtype=self.out_dtype)

        def product_call_in_place(x, out):
            """The product in-place evaluation function."""
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
            """The quotient out-of-place evaluation function."""
            return np.asarray(x1_call_oop(x) / x2_call_oop(x),
                              dtype=self.out_dtype)

        def quotient_call_in_place(x, out):
            """The quotient in-place evaluation function."""
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
            """The power out-of-place evaluation function."""
            if p == 0:
                return self.one()
            elif p == int(p) and p >= 1:
                return np.asarray(pow_posint(x_call_oop(x), int(p)),
                                  dtype=self.out_dtype)
            else:
                return np.power(x_call_oop(x), p).astype(self.out_dtype)

        def power_call_in_place(x, out):
            """The power in-place evaluation function."""
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
            rdtype = _TYPE_MAP_C2R.get(self.out_dtype, None)
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
            rdtype = _TYPE_MAP_C2R.get(self.out_dtype, None)
            rspace = self.astype(rdtype)
            return rspace.element(imagpart_oop)

    @property
    def element_type(self):
        """`FunctionSpaceVector`"""
        return FunctionSpaceVector

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '{!r}'.format(self.domain)
        dtype_str = dtype_repr(self.out_dtype)

        if self.field == RealNumbers():
            if self.out_dtype == np.dtype('float64'):
                pass
            else:
                inner_str += ', out_dtype={}'.format(dtype_str)

        elif self.field == ComplexNumbers():
            if self.out_dtype == np.dtype('complex128'):
                inner_str += ', field={!r}'.format(self.field)
            else:
                inner_str += ', out_dtype={}'.format(dtype_str)

        else:  # different field, name explicitly
            inner_str += ', field={!r}'.format(self.field)
            inner_str += ', out_dtype={}'.format(dtype_str)

        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        inner_str = '{}'.format(self.domain)
        dtype_str = dtype_repr(self.out_dtype)

        if self.field == RealNumbers():
            if self.out_dtype == np.dtype('float64'):
                pass
            else:
                inner_str += ', out_dtype={}'.format(dtype_str)

        elif self.field == ComplexNumbers():
            if self.out_dtype == np.dtype('complex128'):
                inner_str += ', field={!r}'.format(self.field)
            else:
                inner_str += ', out_dtype={}'.format(dtype_str)

        else:  # different field, name explicitly
            inner_str += ', field={!r}'.format(self.field)
            inner_str += ', out_dtype={}'.format(dtype_str)

        return '{}({})'.format(self.__class__.__name__, inner_str)


class FunctionSpaceVector(LinearSpaceVector, FunctionSetVector):

    """Representation of a `FunctionSpace` element."""

    def __init__(self, fspace, fcall):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The set of functions this element lives in
        fcall : `callable`
            The actual instruction for out-of-place evaluation.
            It must return an `FunctionSet.range` element or a
            `numpy.ndarray` of such (vectorized call).
        """
        if not isinstance(fspace, FunctionSpace):
            raise TypeError('function space {!r} not a `FunctionSpace` '
                            'instance.'.format(fspace))

        FunctionSetVector.__init__(self, fspace, fcall)
        LinearSpaceVector.__init__(self, fspace)

    # Tradeoff: either we subclass LinearSpaceVector first and override the
    # 3 methods in FunctionSetVector (as below) which LinearSpaceVector
    # also has, or we switch inheritance order and need to override all magic
    # methods from LinearSpaceVector which are not in-place. This is due to
    # the fact that FunctionSetVector inherits from Operator which defines
    # some of those magic methods, and those do not work in this case.
    __eq__ = FunctionSetVector.__eq__
    assign = FunctionSetVector.assign
    copy = FunctionSetVector.copy

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
        """Function returning the real part of a result."""
        return self.space._realpart(self)

    @property
    def imag(self):
        return self.space._imagpart(self)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
