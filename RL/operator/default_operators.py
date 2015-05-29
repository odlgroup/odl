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
from __future__ import (division, print_function, absolute_import)

from future import standard_library
from builtins import str, super

# RL imports
import RL.operator.operator as op
from RL.space.space import LinearSpace
from RL.space.set import UniversalSet
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
                repr(self._scal) + ')')

    def __str__(self):
        return str(self._scal) + "*I"



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

def instance_method(function):
    """ Adds a self argument to a function 
    such that it may be used as a instance method
    """
    def method(self, *args, **kwargs):
        return function(*args, **kwargs)

    return method

def operator(call=None, apply=None, inv=None, deriv=None, 
             domain=UniversalSet(), range=UniversalSet()):
    """ Creates a simple operator.

    Mostly intended for testing.

    Parameters
    ----------
    call : Function taking one argument (rhs) returns result
           The operators _call method
    apply : Function taking two arguments (rhs, out) returns None
            The operators _apply method
    inv : Operator, optional
          The inverse operator
          Default: None
    deriv : LinearOperator, optional
            The derivative operator
            Default: None
    domain : Set, optional
             The domain of the operator
             Default: UniversalSet
    range : Set, optional
            The range of the operator
            Default: UniversalSet

    Returns
    -------
    operator : Operator
               An operator with the required properties

    Example
    -------
    >>> A = operator(lambda x: 3*x)
    >>> A(5)
    15
    """


    if call is None and apply is None:
        raise ValueError("Need to supply at least one of call or apply")

    metaclass = op.Operator.__metaclass__

    SimpleOperator = metaclass('SimpleOperator',
                               (op.Operator,),
                               {'_call': instance_method(call),
                                '_apply': instance_method(apply),
                                'inverse': inv,
                                'derivative': deriv,
                                'domain': domain,
                                'range': range})

    return SimpleOperator()

def linear_operator(call=None, apply=None, inv=None, deriv=None, adj=None, 
                    domain=UniversalSet(), range=UniversalSet()):
    """ Creates a simple operator.

    Mostly intended for testing.

    Parameters
    ----------
    call : Function taking one argument (rhs) returns result
           The operators _call method
    apply : Function taking two arguments (rhs, out) returns None
            The operators _apply method
    inv : Operator, optional
          The inverse operator
          Default: None
    deriv : LinearOperator, optional
            The derivative operator
            Default: None
    adj : LinearOperator, optional
          The adjoint of the operator
          Defualt: None
    domain : Set, optional
             The domain of the operator
             Default: UniversalSet
    range : Set, optional
            The range of the operator
            Default: UniversalSet

    Returns
    -------
    operator : LinearOperator
               An operator with the required properties

    Example
    -------
    >>> A = linear_operator(lambda x: 3*x)
    >>> A(5)
    15
    """


    if call is None and apply is None:
        raise ValueError("Need to supply at least one of call or apply")

    metaclass = op.LinearOperator.__metaclass__

    SimpleLinearOperator = metaclass('SimpleOperator',
                                     (op.LinearOperator,),
                                     {'_call': instance_method(call),
                                      '_apply': instance_method(apply),
                                      'inverse': inv,
                                      'derivative': deriv,
                                      'domain': domain,
                                      'range': range})

    return SimpleLinearOperator()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
