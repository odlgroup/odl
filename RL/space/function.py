""" Module for spaces whose elements are Functionals
"""
# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from future import standard_library
try:
    from builtins import super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import super

# RL imports
import RL.operator.operator as fun
from RL.space.space import HilbertSpace, Algebra

from RL.space.set import RealNumbers, ComplexNumbers, Set
from RL.utility.utility import errfmt

standard_library.install_aliases()


# Example of a space:
class FunctionSpace(Algebra):
    """ The space of scalar valued functions on some domain

    Parameters
    ----------

    domain : Set
             The set the functions take values from
    field : {RealNumbers, ComplexNumbers}, optional
            The field that the functions map values into.
            Since FunctionSpace is a LinearSpace, this is also
            the set of scalars for this space.
    """

    def __init__(self, domain, field=RealNumbers()):
        if not isinstance(domain, Set):
            raise TypeError("domain ({!r}) is not a Set instance".format(domain))

        if not isinstance(field, (RealNumbers, ComplexNumbers)):
            raise TypeError("field ({!r}) is not a RealNumbers or ComplexNumbers".format(field))

        self.domain = domain
        self._field = field

    def element(self, function=None):
        """ Creates an element in FunctionSpace

        Parameters
        ----------
        function : Function from self.domain to self.field
                   The function that should be converted/reinterpreted as a vector.

        Returns
        -------
        FunctionSpace.Vector instance


        Examples
        --------

        >>> R = RealNumbers()
        >>> space = FunctionSpace(R, R)
        >>> x = space.element(lambda t: t**2)
        >>> x(1)
        1
        >>> x(3)
        9
        """

        if function is None:
            def function(*args):
                return 0
        return FunctionSpace.Vector(self, function)

    def _lincomb(self, z, a, x, b, y):
        """ Returns a function that calculates (a*x + b*y)(t) = a*x(t) + b*y(t)

        The created object is rather slow, and should only be used for testing purposes.
        """
        z._function = a*x + b*y  # Use operator overloading

    def _multiply(self, x, y):
        """ Returns a function that calculates (x * y)(t) = x(t) * y(t)

        The created object is rather slow, and should only be used for testing purposes.
        """
        return self.element(lambda *args: x(*args)*y(*args))

    @property
    def field(self):
        """ The field that the functions map values into.

        Since FunctionSpace is a LinearSpace, this is also
        the set of scalars for this space.
        """
        return self._field

    def equals(self, other):
        """ Verify that other is a FunctionSpace with the same domain and field
        """
        return (isinstance(other, FunctionSpace) and
                self.domain == other.domain and
                self.field == other.field)

    def zero(self):
        """ Returns the zero function (the function which maps any value to zero)
        """
        return self.element(lambda *args: 0)

    def __str__(self):
        if isinstance(self.field, RealNumbers):
            return "FunctionSpace(" + str(self.domain) + ")"
        else:
            return "FunctionSpace(" + str(self.domain) + ", " + str(self.field) + ")"

    def __repr__(self):
        if isinstance(self.field, RealNumbers):
            return "FunctionSpace(" + repr(self.domain) + ")"
        else:
            return "FunctionSpace(" + repr(self.domain) + ", " + repr(self.field) + ")"

    class Vector(Algebra.Vector, fun.Operator):
        """ A Vector in a FunctionSpace

        FunctionSpace-Vectors are themselves also Functionals, and inherit
        a large set of features from them.

        Parameters
        ----------

        space : FunctionSpace
            Instance of FunctionSpace this vector lives in
        function : Function from space.domain to space.field
            The function that should be converted/reinterpreted as a vector.
        """

        def __init__(self, space, function):
            super().__init__(space)
            if not callable(function):
                raise TypeError("'function' is not callable")
            self._function = function

        def _call(self, rhs):
            """ Apply the functional in some point
            """
            return self._function(rhs)

        @property
        def domain(self):
            """ The range of this Vector (when viewed as a functional)
            """
            return self.space.domain

        @property
        def range(self):
            """ The range of this Vector (when viewed as a functional)

            The range is the same as the field of the vectors space
            """
            return self.space.field

        def __str__(self):
            return str(self._function)

        def __repr__(self):
            return repr(self.space) + '.element(' + repr(self._function) + ')'


class L2(FunctionSpace, HilbertSpace):
    """The space of square integrable functions on some domain
    """

    def __init__(self, domain, field=RealNumbers()):
        super().__init__(domain, field)

    def _inner(self, v1, v2):
        """ TODO: remove?
        """
        raise NotImplementedError(errfmt('''
        You cannot calculate inner products in non-discretized spaces'''))

    def equals(self, other):
        """ Verify that other is equal to this space as a FunctionSpace and also a L2 space.
        """
        return isinstance(other, L2) and FunctionSpace.equals(self, other)

    def __str__(self):
        if isinstance(self.field, RealNumbers):
            return "L2(" + str(self.domain) + ")"
        else:
            return "L2(" + str(self.domain) + ", " + str(self.field) + ")"

    def __repr__(self):
        if isinstance(self.field, RealNumbers):
            return "L2(" + repr(self.domain) + ")"
        else:
            return "L2(" + repr(self.domain) + ", " + repr(self.field) + ")"

    class Vector(FunctionSpace.Vector, HilbertSpace.Vector):
        """ A Vector in a L2-space

        FunctionSpace-Vectors are themselves also Functionals, and inherit
        a large set of features from them.

        Parameters
        ----------

        space : FunctionSpace
            Instance of FunctionSpace this vector lives in
        function : Function from space.domain to space.field
            The function that should be converted/reinterpreted as a vector.
        """

if __name__ == '__main__':
    import doctest
    doctest.testmod()
