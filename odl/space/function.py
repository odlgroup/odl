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

"""Spaces of functions with common domain and range.

TODO: document properly
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import super

# External imports
import numpy as np

# ODL imports
from odl.operator.operator import Operator
from odl.set.domain import IntervalProd
from odl.set.set import RealNumbers, ComplexNumbers, Set
from odl.set.space import LinearSpace


__all__ = ('FunctionSet', 'FunctionSpace')


class FunctionSet(Set):

    """A general set of functions with common domain and range."""

    def __init__(self, dom, ran):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `Set`
            The domain of the functions.
        ran : `Set`
            The range of the functions.
        """
        if not isinstance(dom, Set):
            raise TypeError('domain {} not a `Set` instance.'.format(dom))

        if not isinstance(ran, Set):
            raise TypeError('range {} not a `Set` instance.'.format(dom))

        self._domain = dom
        self._range = ran

    @property
    def domain(self):
        """Return `domain` attribute."""
        return self._domain

    @property
    def range(self):
        """Return `range` attribute."""
        return self._range

    def element(self, fcall=None, fapply=None):
        """Create a `FunctionSet` element.

        Parameters
        ----------
        fcall : callable, optional
            The actual instruction for out-of-place evaluation.
            It must return an `fset.range` element or a
            `numpy.ndarray` of such (vectorized call).

            If `fcall` is a `FunctionSet.Vector`, it is wrapped
            as a new `Vector`.

        fapply : callable, optional
            The actual instruction for in-place evaluation.
            Its first argument must be the `fset.range` element
            or the array of such (vectorization) to which the
            result is written.

            If `fapply` is a `FunctionSet.Vector`, it is wrapped
            as a new `Vector`.

        *At least one of the arguments `fcall` and `fapply` must
        be provided.*

        Returns
        -------
        element : `FunctionSet.Vector`
            The new element created from `func`
        """
        if isinstance(fcall, self.Vector):  # no double wrapping
            return self.element(fcall._call, fcall._apply)
        elif isinstance(fapply, self.Vector):
            return self.element(fapply._call, fapply._apply)
        else:
            return self.Vector(self, fcall, fapply)

    def equals(self, other):
        """Test if `other` is equal to this set.

        Returns
        -------
        equals : `bool`
            `True` if `other` is a `FunctionSet` with same `domain`
            and `range`, `False` otherwise.
        """
        return (type(self) == type(other) and
                self.domain == other.domain and
                self.range == other.range)

    def contains(self, other):
        """Test if `other` is contained in this set.

        Returns
        -------
        equals : `bool`
            `True` if `other` is a `FunctionSet.Vector` whose `space`
            attribute equals this space, `False` otherwise.
        """
        return (isinstance(other, FunctionSet.Vector) and
                self == other.space)

    def __repr__(self):
        """`s.__repr__() <==> repr(s)`."""
        return 'FunctionSet({!r}, {!r})'.format(self.domain, self.range)

    def __str__(self):
        """`s.__str__() <==> str(s)`."""
        return 'FunctionSet({}, {})'.format(self.domain, self.range)

    class Vector(Operator):

        """Representation of a `FunctionSet` element."""

        def __init__(self, fset, fcall=None, fapply=None):
            """Initialize a new instance.

            Parameters
            ----------
            fset : `FunctionSet`
                The set of functions this element lives in
            fcall : callable, optional
                The actual instruction for out-of-place evaluation.
                It must return an `fset.range` element or a
                `numpy.ndarray` of such (vectorized call).
            fapply : callable, optional
                The actual instruction for in-place evaluation.
                Its first argument must be the `fset.range` element
                or the array of such (vectorization) to which the
                result is written.

            *At least one of the arguments `fcall` and `fapply` must
            be provided.*
            """
            if not isinstance(fset, FunctionSet):
                raise TypeError('function set {} not a `FunctionSet` '
                                'instance.'.format(fset))

            if fcall is None and fapply is None:
                raise ValueError('call function and apply function cannot '
                                 'both be `None`.')

            if fcall is not None and not callable(fcall):
                raise TypeError('call function {} is not callable.'
                                ''.format(fcall))

            if fapply is not None and not callable(fapply):
                raise TypeError('apply function {} is not callable.'
                                ''.format(fapply))

            self._space = fset
            if fcall is not None:
                self._call = fcall
            if fapply is not None:
                self._apply = fapply

        @property
        def space(self):
            """Return `space` attribute."""
            return self._space

        @property
        def domain(self):
            """The function domain (abstract in `Operator`)."""
            return self.space.domain

        @property
        def range(self):
            """The function range (abstract in `Operator`)."""
            return self.space.range

        def equals(self, other):
            """Test `other` for equality.

            Returns
            -------
            equals : `bool`
                `True` if `other` is a `FunctionSet.Vector` with
                `other.space` equal to this vector's space and
                the call and apply implementations of `other` and
                this vector are equal. `False` otherwise.
            """
            return (isinstance(other, FunctionSet.Vector) and
                    self.space == other.space and
                    self._call == other._call and
                    self._apply == other._apply)

        def __call__(self, *inp):
            """Vectorized and multi-argument out-of-place evaluation.

            Parameters
            ----------
            inp1,...,inpN : `object`
                Input arguments for the function evaluation.

            Returns
            -------
            outp : `range` element or array of elements
                Result of the function evaluation.

            Raises
            ------
            If `outp` is not a `range` element or a `numpy.ndarray`
            with `outp[0] in range`, a `TypeError` is raised.
            """
            if inp in self.domain:
                # single value list: f(0, 1, 2)
                pass
            elif inp[0] in self.domain:
                # single array: f([0, 1, 2])
                pass
            elif isinstance(self.domain, IntervalProd):
                # Vectorization only allowed in this case

                # First case: (N, d) array of points, where d = dimension
                if (isinstance(inp[0], np.ndarray) and
                        inp[0].ndim == 2 and
                        inp[0].shape[1] == self.domain.dim):
                    min_coords = np.min(inp[0], axis=0)
                    max_coords = np.max(inp[0], axis=0)

                # Second case: d meshgrid type arrays
                elif (len(inp) == self.domain.dim and
                      all(isinstance(vec, np.ndarray) for vec in inp)):
                    min_coords = [np.min(vec) for vec in inp]
                    max_coords = [np.max(vec) for vec in inp]

                if (min_coords not in self.domain or
                        max_coords not in self.domain):
                    raise ValueError('input contains points outside '
                                     '`domain` {}.'.format(self.domain))
            else:
                raise TypeError('input is neither an element of the function '
                                'domain {} nor an array or meshgrid-type '
                                'coordinate list.'.format(self.domain))

            outp = self._call(*inp)

            if not (outp in self.range or
                    (isinstance(outp, np.ndarray) and
                     outp.flat[0] in self.range)):
                raise TypeError('result {!r} not an element or an array of '
                                'elements of the function range {}.'
                                ''.format(outp, self.range))

            return outp

        def _apply(self, outp, *inp):
            """Vectorized and multi-argument in-place evaluation.

            Parameters
            ----------
            outp : `range` element or array of elements
                Element(s) to which the result is written.
            inp1,...,inpN : `object`
                Input arguments for the function evaluation.

            Returns
            -------
            None

            Raises
            ------
            If `outp` is not a `range` element or a `numpy.ndarray`
            with `outp[0] in range`, a `TypeError` is raised.
            """
            if not (outp in self.range or
                    (isinstance(outp, np.ndarray) and
                     outp.flat[0] in self.range)):
                raise TypeError('result {!r} not an element or an array of '
                                'elements of the function range {}.'
                                ''.format(outp, self.range))

            return self._apply(outp, *inp)

        def __eq__(self, other):
            """`vec.__eq__(other) <==> vec == other`"""
            return self.equals(other)

        def __ne__(self, other):
            """`vec.__ne__(other) <==> vec != other`"""
            return not self.equals(other)

        def __str__(self):
            return str(self._call)

        def __repr__(self):
            return '{!r}.element({!r})'.format(self.space, self._call)


class FunctionSpace(FunctionSet, LinearSpace):

    """A vector space of functions."""

    def __init__(self, dom, field):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `Set`
            The domain of the functions.
        field : `RealNumbers` or `ComplexNumbers`
            The range of the functions.
        """
        if not isinstance(dom, Set):
            raise TypeError('domain {} not a `Set` instance.'.format(dom))

        if not (isinstance(field, RealNumbers) or
                isinstance(field, ComplexNumbers)):
            raise TypeError('field {} not a `RealNumbers` or `ComplexNumbers` '
                            'instance.'.format(field))

        super().__init__(dom, field)
        self._field = field

    @property
    def field(self):
        """Return `field` attribute."""
        return self._field

    def element(self, fcall=None, fapply=None):
        """Create a `FunctionSet` element.

        Parameters
        ----------
        fcall : callable, optional
            The actual instruction for out-of-place evaluation.
            It must return an `fset.range` element or a
            `numpy.ndarray` of such (vectorized call).

            If `fcall` is a `FunctionSet.Vector`, it is wrapped
            as a new `FunctionSpace.Vector`.

        fapply : callable, optional
            The actual instruction for in-place evaluation.
            Its first argument must be the `fset.range` element
            or the array of such (vectorization) to which the
            result is written.

            If `fapply` is a `FunctionSet.Vector`, it is wrapped
            as a new `FunctionSpace.Vector`.

        Returns
        -------
        `element` : `FunctionSpace.Vector`
            The new element.
        """
        if fcall is None and fapply is None:
            return self.zero()
        elif isinstance(fcall, FunctionSet.Vector):  # no double wrapping
            return self.element(fcall._call, fcall._apply)
        elif isinstance(fapply, FunctionSet.Vector):
            return self.element(fapply._call, fapply._apply)
        else:
            return self.Vector(self, fcall, fapply)

    def _lincomb(self, z, a, x, b, y):
        """Raw linear combination of `x` and `y`.

        Note
        ----
        The additions and multiplications are implemented via a simple
        Python function, so the resulting function is probably slow.
        """
        # Store to allow aliasing
        x_old_call = x._call
        x_old_apply = x._apply
        y_old_call = y._call
        y_old_apply = y._apply

        def lincomb_call(*inp):
            """Linear combination, call version."""
            # Due to vectorization, at least one call must be made to
            # ensure the correct final shape. The rest is optimized as
            # far as possible.
            if a == 0 and b != 0:
                outp = y_old_call(*inp)
                if b != 1:
                    outp *= b
            elif b == 0:  # Contains the case a == 0
                outp = x_old_call(*inp)
                if a != 1:
                    outp *= a
            else:
                outp = x_old_call(*inp)
                if a != 1:
                    outp *= a
                tmp = y_old_call(*inp)
                if b != 1:
                    tmp *= b
                outp += tmp

            return outp

        def lincomb_apply(outp, *inp):
            """Linear combination, apply version."""
            # TODO: allow also CudaRn-like container types
            if not isinstance(outp, np.ndarray):
                raise TypeError('in-place evaluation only possible if output '
                                'is of type `numpy.ndarray`.')
            if a == 0 and b == 0:
                outp *= 0
            elif a == 0 and b != 0:
                y_old_apply(outp, *inp)
                if b != 1:
                    outp *= b
            elif b == 0 and a != 0:
                x_old_apply(outp, *inp)
                if a != 1:
                    outp *= a
            else:
                tmp = np.empty_like(outp)
                x_old_apply(outp, *inp)
                y_old_apply(tmp, *inp)
                if a != 1:
                    outp *= a
                if b != 1:
                    tmp *= b

                outp += tmp

        z._call = lincomb_call
        z._apply = lincomb_apply

    def zero(self):
        """The function mapping everything to zero.

        Since `lincomb` is slow, we implement this function directly.
        """
        def zero_(*_):
            """The zero function."""
            return self.field.element(0.0)
        return self.element(zero_)

    def equals(self, other):
        """Test if `other` is equal to this space.

        Returns
        -------
        equals : `bool`
            `True` if `other` is a `FunctionSpace` with same `domain`
            and `range`, `False` otherwise.
        """
        return (isinstance(other, FunctionSpace) and
                self.domain == other.domain and
                self.range == other.range)

    def _multiply(x, y):
        """Raw pointwise multiplication of two functions.

        Note
        ----
        The multiplication is implemented with a simple Python
        function, so the resulting function object is probably slow.
        """
        x_old = x.function
        y_old = y.function

        def product(arg):
            """The actual product function."""
            return x_old(arg) * y_old(arg)
        y._function = product

    class Vector(FunctionSet.Vector, LinearSpace.Vector):

        """Representation of a `FunctionSpace` element."""

        def __init__(self, fspace, fcall=None, fapply=None):
            """Initialize a new instance.

            Parameters
            ----------
            fspace : `FunctionSpace`
                The set of functions this element lives in
            fcall : callable, optional
                The actual instruction for out-of-place evaluation.
                It must return an `fset.range` element or a
                `numpy.ndarray` of such (vectorized call).
            fapply : callable, optional
                The actual instruction for in-place evaluation.
                Its first argument must be the `fset.range` element
                or the array of such (vectorization) to which the
                result is written.

            *At least one of the arguments `fcall` and `fapply` must
            be provided.*
            """
            if not isinstance(fspace, FunctionSpace):
                raise TypeError('function space {} not a `FunctionSpace` '
                                'instance.'.format(fspace))

            super().__init__(fspace, fcall, fapply)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
