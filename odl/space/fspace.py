# Copyright 2014, 2015 The ODL development group
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
from odl.util.utility import preload_call_with, preload_default_oop_call_with
from odl.util.vectorization import (
    is_valid_input_array, is_valid_input_meshgrid,
    meshgrid_input_order, out_shape_from_array, out_shape_from_meshgrid,
    vectorize)


__all__ = ('FunctionSet', 'FunctionSetVector',
           'FunctionSpace', 'FunctionSpaceVector')


def _default_in_place(func, x, out, **kwargs):
    """Default in-place evaluation method."""
    out[:] = func(x, **kwargs)
    return out


def _default_out_of_place(func, dtype, x, **kwargs):
    """Default in-place evaluation method."""
    if is_valid_input_array(x, func.domain.ndim):
        out_shape = out_shape_from_array(x)
    elif is_valid_input_meshgrid(x, func.domain.ndim):
        out_shape = out_shape_from_meshgrid(x)
    else:
        raise TypeError('cannot use in-place method to implement '
                        'out-of-place non-vectorized evaluation.')

    # TODO: implement this. Needs a helper to infer data type without
    # creating an array.
    if dtype is None:
        raise NotImplementedError('lazy out-of-place default not implemented.')

    out = np.empty(out_shape, dtype=dtype)
    func(x, out=out, **kwargs)
    return out


class FunctionSet(Set):

    """A general set of functions with common domain and range."""

    def __init__(self, domain, range):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `Set`
            The domain of the functions.
        range : `Set`
            The range of the functions.
        """
        if not isinstance(domain, Set):
            raise TypeError('domain {!r} not a `Set` instance.'.format(domain))

        if not isinstance(range, Set):
            raise TypeError('range {!r} not a `Set` instance.'.format(range))

        self._domain = domain
        self._range = range

    @property
    def domain(self):
        """Common domain of all functions in this set."""
        return self._domain

    @property
    def range(self):
        """Common range of all functions in this set."""
        return self._range

    def element(self, fcall=None, vectorized=True):
        """Create a `FunctionSet` element.

        Parameters
        ----------
        fcall : `callable`, optional
            The actual instruction for out-of-place evaluation.
            It must return an `range` element or a
            `numpy.ndarray` of such (vectorized call).

        vectorized : bool
            Whether ``fcall`` supports vectorized evaluation.

        Returns
        -------
        element : `FunctionSetVector`
            The new element, always supports vectorization

        See also
        --------
        TensorGrid.meshgrid : efficient grids for function
            evaluation
        """
        if not callable(fcall):
            raise TypeError('function {!r} is not callable.'.format(fcall))

        if not vectorized:
            fcall = vectorize(dtype=None)(fcall)

        return self.element_type(self, fcall)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if ``other`` is a `FunctionSet` with same
            `FunctionSet.domain` and `FunctionSet.range`,
            `False` otherwise.
        """
        if other is self:
            return True

        return (isinstance(other, FunctionSet) and
                self.domain == other.domain and
                self.range == other.range)

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
        """ `FunctionSetVector` """
        return FunctionSetVector


class FunctionSetVector(Operator):

    """Representation of a `FunctionSet` element."""

    def __init__(self, fset, fcall):
        """Initialize a new instance.

        Parameters
        ----------
        fset : `FunctionSet`
            The set of functions this element lives in
        fcall : `callable`
            The actual instruction for out-of-place evaluation.
            It must return an `FunctionSet.range` element or a
            `numpy.ndarray` of such (vectorized call).
        """
        self._space = fset
        super().__init__(self._space.domain, self._space.range, linear=False)

        # Determine which type of implementation fcall is
        if isinstance(fcall, FunctionSetVector):
            call_has_out, call_out_optional, _ = _dispatch_call_args(
                bound_call=fcall._call)
        elif isinstance(fcall, np.ufunc):
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
            self._call_in_place = preload_call_with(self, 'ip')(
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
            # Lazy out-of-place evaluation is not implemented yet.
            self._call_out_of_place = preload_default_oop_call_with(self)(
                _default_out_of_place)

    @property
    def space(self):
        """The space or set this function belongs to."""
        return self._space

    def _call(self, x, out=None, **kwargs):
        """Raw evaluation method."""
        if out is None:
            out = self._call_out_of_place(x, **kwargs)
        else:
            self._call_in_place(x, out=out, **kwargs)
        return out

    def __call__(self, x, out=None, **kwargs):
        """Out-of-place evaluation.

        Parameters
        ----------
        x : object
            Input argument for the function evaluation. Conditions
            on `x` depend on vectorization:

            `False` : ``x`` must be a domain element

            `True` : ``x`` must be a `numpy.ndarray` with shape
            ``(d, N)``, where ``d`` is the number of dimensions of
            the function domain
            OR
            `x` is a sequence of `numpy.ndarray` with length
            `space.ndim`, and the arrays can be broadcast
            against each other.

        out : `numpy.ndarray`, optional
            Output argument holding the result of the function
            evaluation, can only be used for vectorized
            functions. Its shape must be equal to
            `np.broadcast(*x).shape`.
            If `out` is given, it is returned.

        kwargs : {'vec_bounds_check'}
            'bounds_check' : bool
                Whether or not to check if all input points lie in
                the function domain. For vectorized evaluation,
                this requires the domain to implement
                `contains_all`.

                Default: `True`

        Returns
        -------
        out : range element or array of elements
            Result of the function evaluation

        Raises
        ------
        TypeError
            If `x` is not a valid vectorized evaluation argument

            If `out` is not a range element or a `numpy.ndarray`
            of range elements

        ValueError
            If evaluation points fall outside the valid domain
        """
        vec_bounds_check = kwargs.pop('vec_bounds_check', True)
        if vec_bounds_check and not hasattr(self.domain, 'contains_all'):
            raise AttributeError('vectorized bounds check not possible for '
                                 'domain {}, missing `contains_all()` '
                                 'method.'.format(self.domain))

        # Check for input type and determine output shape
        if is_valid_input_array(x, self.domain.ndim):
            out_shape = out_shape_from_array(x)
            scalar_out = False
            # For 1d, squeeze the array
            if self.domain.ndim == 1 and x.ndim == 2:
                x = x[0]
        elif is_valid_input_meshgrid(x, self.domain.ndim):
            out_shape = out_shape_from_meshgrid(x)
            scalar_out = False
            # For 1d, fish out the vector from the tuple
            if self.domain.ndim == 1:
                x = x[0]
        elif x in self.domain:
            x = np.atleast_2d(x).T  # make a (d, 1) array
            out_shape = (1,)
            scalar_out = (out is None)
        else:
            # Unknown input
            raise TypeError('argument {!r} not a valid vectorized '
                            'input. Expected an element of the domain '
                            '{dom}, a ({dom.ndim}, n) array '
                            'or a length-{dom.ndim} meshgrid sequence.'
                            ''.format(x, dom=self.domain))

        # Check bounds if specified
        if vec_bounds_check:
            if not self.domain.contains_all(x):
                raise ValueError('input contains points outside '
                                 'the domain {}.'.format(self.domain))

        # Call the function and check out shape, before or after
        if out is None:
            out = self._call(x, **kwargs)
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
            self._call(x, out=out, **kwargs)

        # Check output values
        if vec_bounds_check:
            if not self.range.contains_all(out):
                raise ValueError('output contains points outside '
                                 'the range {}.'
                                 ''.format(self.domain))

        return out[0] if scalar_out else out

    def assign(self, other):
        """Assign `other` to this vector.

        This is implemented without `lincomb` to ensure that
        `vec == other` evaluates to `True` after
        `vec.assign(other)`.
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
        """`vec.__eq__(other) <==> vec == other`.

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

    def __init__(self, domain, field=RealNumbers()):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `Set`
            The domain of the functions
        field : `Field`, optional
            The range of the functions.
        """
        if not isinstance(domain, Set):
            raise TypeError('domain {!r} not a Set instance.'.format(domain))

        if not isinstance(field, Field):
            raise TypeError('field {!r} not a `Field` instance.'
                            ''.format(field))

        FunctionSet.__init__(self, domain, field)
        LinearSpace.__init__(self, field)

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

        Note
        ----
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

                fcall = vectorize(dtype=dtype)(fcall)

            return self.element_type(self, fcall)

    def zero(self):
        """The function mapping everything to zero.

        This function is the additive unit in the function space.

        Since `lincomb` may be slow, we implement this function
        directly.
        """
        dtype = complex if self.field == ComplexNumbers() else float

        def zero_vec(x, out=None):
            """The zero function, vectorized."""
            if is_valid_input_meshgrid(x, self.domain.ndim):
                order = meshgrid_input_order(x)
            else:
                order = 'C'

            out_shape = out_shape_from_meshgrid(x)
            if out is None:
                return np.zeros(out_shape, dtype=dtype, order=order)
            else:
                out.fill(0)

        return self.element_type(self, zero_vec)

    def one(self):
        """The function mapping everything to one.

        This function is the multiplicative unit in the function space.
        """
        dtype = complex if self.field == ComplexNumbers() else float

        def one_vec(x, out=None):
            """The one function, vectorized."""
            if is_valid_input_meshgrid(x, self.domain.ndim):
                order = meshgrid_input_order(x)
            else:
                order = 'C'

            out_shape = out_shape_from_meshgrid(x)
            if out is None:
                return np.ones(out_shape, dtype=dtype, order=order)
            else:
                out.fill(1)

        return self.element_type(self, one_vec)

    def __eq__(self, other):
        """`s.__eq__(other) <==> s == other`.

        Returns
        -------
        equals : `bool`
            `True` if `other` is a `FunctionSpace` with same `domain`
            and `range`, `False` otherwise.
        """
        if other is self:
            return True

        return (isinstance(other, FunctionSpace) and
                FunctionSet.__eq__(self, other))

    def _lincomb(self, a, x1, b, x2, out):
        """Raw linear combination of `x1` and `x2`.

        Note
        ----
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
                out = x2_call_oop(x)
                if b != 1:
                    out *= b
            elif b == 0:  # Contains the case a == 0
                out = x1_call_oop(x)
                if a != 1:
                    out *= a
            else:
                out = x1_call_oop(x)
                if a != 1:
                    out *= a
                tmp = x2_call_oop(x)
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
        x1_call_oop = x1._call_out_of_place
        x1_call_ip = x1._call_in_place
        x2_call_oop = x2._call_out_of_place
        x2_call_ip = x2._call_in_place

        def product_call_out_of_place(x):
            """The product out-of-place evaluation function."""
            return x1_call_oop(x) * x2_call_oop(x)

        def product_call_in_place(x, out):
            """The product in-place evaluation function."""
            tmp = np.empty_like(out)
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
        x1_call_oop = x1._call_out_of_place
        x1_call_ip = x1._call_in_place
        x2_call_oop = x2._call_out_of_place
        x2_call_ip = x2._call_in_place

        def quotient_call_out_of_place(x):
            """The quotient out-of-place evaluation function."""
            return x1_call_oop(x) / x2_call_oop(x)

        def quotient_call_in_place(x, out):
            """The quotient in-place evaluation function."""
            tmp = np.empty_like(out)
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
            if p == int(p) and p >= 1:
                return pow_posint(x_call_oop(x), int(p))
            else:
                return x_call_oop(x) ** p

        def power_call_in_place(x, out):
            """The power in-place evaluation function."""
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

    @property
    def element_type(self):
        """`FunctionSpaceVector`"""
        return FunctionSpaceVector


class FunctionSpaceVector(FunctionSetVector, LinearSpaceVector):

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

    # Some additional magic methods not defined for arbitrary linear spaces
    def __add__(self, other):
        """Return ``self + other``."""
        if other in self.space.field:
            # other --> other * space.one()
            tmp = self.space.one()
            self.space.lincomb(other, tmp, out=tmp)
            return self.space.lincomb(1, self, 1, tmp, out=tmp)
        elif other in self.space:
            return self.space.lincomb(1, self, 1, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        """Return ``other + self``."""
        if other in self.space.field:
            # other --> other * space.one()
            tmp = self.space.one()
            self.space.lincomb(other, tmp, out=tmp)
            return self.space.lincomb(1, tmp, 1, self, out=tmp)
        elif other in self.space:
            return self.space.lincomb(1, other, 1, self)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Return ``self - other``."""
        if other in self.space.field:
            # other --> other * space.one()
            tmp = self.space.one()
            self.space.lincomb(other, tmp, out=tmp)
            return self.space.lincomb(1, self, -1, tmp, out=tmp)
        elif other in self.space:
            return self.space.lincomb(1, self, -1, other)
        else:
            return NotImplemented

    def __rsub__(self, other):
        """Return ``other - self``."""
        if other in self.space.field:
            # other --> other * space.one()
            tmp = self.space.one()
            self.space.lincomb(other, tmp, out=tmp)
            return self.space.lincomb(1, tmp, -1, self, out=tmp)
        elif other in self.space:
            return self.space.lincomb(1, other, -1, self)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Return ``self * other``."""
        if other in self.space.field:
            # other --> other * space.one()
            tmp = self.space.one()
            self.space.lincomb(other, tmp, out=tmp)
            return self.space.multiply(self, tmp, out=tmp)
        elif other in self.space:
            return self.space.multiply(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """Return ``other * self``."""
        if other in self.space.field:
            # other --> other * space.one()
            tmp = self.space.one()
            self.space.lincomb(other, tmp, out=tmp)
            return self.space.multiply(tmp, self, out=tmp)
        elif other in self.space:
            return self.space.multiply(other, self)
        else:
            return NotImplemented

    def __truediv__(self, other):
        """Return ``self / other``."""
        if other in self.space.field:
            # other --> other * space.one()
            tmp = self.space.one()
            self.space.lincomb(other, tmp, out=tmp)
            return self.space.divide(self, tmp, out=tmp)
        elif other in self.space:
            return self.space.divide(self, other)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        """Return ``other / self``."""
        if other in self.space.field:
            # other --> other * space.one()
            tmp = self.space.one()
            self.space.lincomb(other, tmp, out=tmp)
            return self.space.divide(tmp, self, out=tmp)
        elif other in self.space:
            return self.space.divide(other, self)
        else:
            return NotImplemented

    __rdiv__ = __rtruediv__

    def __pow__(self, p):
        """`f.__pow__(p) <==> f ** p`."""
        out = self.space.element()
        self.space._scalar_power(self, p, out=out)
        return out

    def __ipow__(self, p):
        """`f.__ipow__(p) <==> f **= p`."""
        return self.space._scalar_power(self, p, out=self)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
