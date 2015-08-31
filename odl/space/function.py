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

"""
Support for functionspaces, such as L2.
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from future import standard_library
from builtins import super

# ODL imports
import odl.operator.operator as fun
from odl.space.space import LinearSpace
from odl.space.set import RealNumbers, ComplexNumbers, Set
from odl.utility.utility import errfmt

standard_library.install_aliases()


# Example of a space:
class FunctionSpace(LinearSpace):
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
            raise TypeError(errfmt('''
            domain ({!r}) is not a Set instance'''.format(domain)))

        if not isinstance(field, (RealNumbers, ComplexNumbers)):
            raise TypeError(errfmt('''
            field ({!r}) is not a RealNumbers or ComplexNumbers.
            '''.format(field)))

        self.domain = domain
        self._field = field

    def element(self, funct=None):
        """ Creates an element in FunctionSpace

        Parameters
        ----------
        funct : Function from self.domain to self.field
            The function that should be converted/reinterpreted
            as a vector.

        Returns
        -------
        FunctionSpace.Vector instance


        Examples
        --------

        >>> R = RealNumbers()
        >>> space = FunctionSpace(R, R)
        >>> x = space.element(lambda t: t**2)
        >>> x(1)
        1.0
        >>> x(3)
        9.0
        """

        if funct is None:
            def function(*_):
                """ A function that always returns zero
                """
                return 0
            funct = function
        return FunctionSpace.Vector(self, funct)

    def _lincomb(self, z, a, x, b, y):
        """ Returns a function that calculates (a*x + b*y)(t) = a*x(t) + b*y(t)

        The created object is rather slow,
        and should only be used for testing purposes.
        """
        # Use operator overloading
        # pylint: disable=protected-access
        z._function = lambda *args: a*x._function(*args) + b*y._function(*args)

    def _multiply(self, x, y):
        """ Returns a function that calculates (x * y)(t) = x(t) * y(t)

        The created object is rather slow,
        and should only be used for testing purposes.
        """
        # pylint: disable=protected-access
        tmp = y._function
        y._function = lambda *args: x._function(*args)*tmp._function(*args)

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
        return (type(self) == type(other) and
                self.domain == other.domain and
                self.field == other.field)

    def zero(self):
        """ Returns the zero function
        The function which maps any value to zero
        """
        return self.element(lambda *args: 0)

    def __str__(self):
        if isinstance(self.field, RealNumbers):
            return "FunctionSpace(" + str(self.domain) + ")"
        else:
            return ("FunctionSpace(" + str(self.domain) + ", " +
                    str(self.field) + ")")

    def __repr__(self):
        if isinstance(self.field, RealNumbers):
            return "FunctionSpace(" + repr(self.domain) + ")"
        else:
            return ("FunctionSpace(" + repr(self.domain) + ", " +
                    repr(self.field) + ")")

    class Vector(LinearSpace.Vector, fun.Operator):
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
            return float(self._function(rhs))

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


class L2(FunctionSpace):
    """The space of square integrable functions on some domain
    """

    def __init__(self, domain, field=RealNumbers()):
        super().__init__(domain, field)

    def _inner(self, v1, v2):
        """ TODO: remove?
        """
        raise NotImplementedError(errfmt('''
        You cannot calculate inner products in non-discretized spaces'''))

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

    class Vector(FunctionSpace.Vector):
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
