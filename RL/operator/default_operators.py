# Copyright 2014, 2015 Jonas Adler
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

"""
Default operators defined on any space

Scale vector by scalar, Identity operation
"""


# Imports for common Python 2/3 codebase
from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import

from future import standard_library

try:
    from builtins import str, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, super

# RL imports
import RL.operator.operator as op
from RL.space.space import LinearSpace
from RL.utility.utility import errfmt

standard_library.install_aliases()


class ScalingOperator(op.SelfAdjointOperator):
    """
    Operator that scales a vector by a scalar

    Parameters
    ----------
    space : LinearSpace
            The space the vectors should lie in
    scalar : space.field element
             An element in the field of the space that
             the vectors should be scaled by
    """
    def __init__(self, space, scalar):
        if not isinstance(space, LinearSpace):
            raise TypeError(errfmt('''
            'space' ({}) must be a LinearSpace instance
            '''.format(space)))

        self._space = space
        self._scal = float(scalar)

    def _apply(self, input, out):
        """
        Scales a vector and stores the result in another

        Parameters
        ----------
        input : self.domain element
                An element in the domain of this operator
        scalar : self.range element
                 An element in the range of this operator

        Returns
        -------
        None

        Example
        -------
        >>> from RL.space.euclidean import RN
        >>> r3 = RN(3)
        >>> vec = r3.element([1, 2, 3])
        >>> out = r3.element()
        >>> op = ScalingOperator(r3, 2.0)
        >>> op.apply(vec, out)
        >>> out
        RN(3).element([ 2.,  4.,  6.])
        """
        out.lincomb(self._scal, input)

    def _call(self, input):
        """
        Scales a vector

        Parameters
        ----------
        input : self.domain element
                An element in the domain of this operator


        Returns
        -------
        scaled : self.range element
                 An element in the range of this operator,
                 input * self.scale

        Example
        -------
        >>> from RL.space.euclidean import RN
        >>> r3 = RN(3)
        >>> vec = r3.element([1, 2, 3])
        >>> op = ScalingOperator(r3, 2.0)
        >>> op(vec)
        RN(3).element([ 2.,  4.,  6.])
        """

        return self._scal * input

    @property
    def inverse(self):
        """
        The inverse of a scaling is scaling by 1/self.scale

        Parameters
        ----------
        None

        Returns
        -------
        inverse : ScalingOperator
                  Scaling by 1/self.scale

        Example
        -------
        >>> from RL.space.euclidean import EuclideanSpace
        >>> r3 = EuclideanSpace(3)
        >>> vec = r3.element([1, 2, 3])
        >>> op = ScalingOperator(r3, 2.0)
        >>> inv = op.inverse
        >>> inv(op(vec)) == vec
        True
        >>> op(inv(vec)) == vec
        True
        """
        return ScalingOperator(self._space, 1.0/self._scal)

    @property
    def domain(self):
        return self._space

    @property
    def range(self):
        return self._space

    def __repr__(self):
        return ('LinCombOperator(' + repr(self._space) + ", " +
                repr(self._scale) + ')')

    def __str__(self):
        return str(self._scale) + "*I"



class IdentityOperator(ScalingOperator):
    """
    The identity operator on a space, copies a vector into another

    Parameters
    ----------

    space : LinearSpace
            The space the vectors should lie in
    """
    def __init__(self, space):
        super().__init__(space, 1)

    def __repr__(self):
        return 'IdentityOperator(' + repr(self._space) + ')'

    def __str__(self):
        return "I"

if __name__ == '__main__':
    import doctest
    doctest.testmod()
