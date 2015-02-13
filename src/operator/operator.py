# -*- coding: utf-8 -*-
"""
operator.py -- functional analytic operators

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from builtins import object
from future import standard_library
standard_library.install_aliases()


class Operator(object):
    """Basic class for a functional analytic operator.
    TODO: write some more
    """

    def __init__(self, map_, **kwargs):

        self._map = map_
        self._left = kwargs.get('left', None)
        self._right = kwargs.get('right', None)

    @property
    def map_(self):
        return self._map

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, other):
        self._right = other
        other._left = self

    def _rightmost(self):
        cur_op = self
        while cur_op.right is not None:
            cur_op = cur_op.right
        return cur_op

    def __call__(self, function):
        cur_op = self._rightmost()
        cur_func = cur_op.map_(function)
        while cur_op.left is not None:
            cur_op = cur_op.left
            cur_func = cur_op.map_(cur_func)
        return cur_func

    def __mul__(self, other):
        if isinstance(other, Operator):  # return operator product
            new_op = self.copy()
            new_op.__imul__(other)
            return new_op
        else:  # try to evaluate
            return self.__call__(other)

    def __imul__(self, other):
        if not isinstance(other, Operator):
            raise TypeError("`other` must be of `Operator` type")
        self._rightmost().right = other

    def copy(self):
        new_op = Operator(self.map_)
        cur_op = self
        cur_cpy = new_op
        while cur_op.right is not None:
            cur_op = cur_op.right
            cur_cpy.right = Operator(cur_op.map_, left=cur_cpy)
            cur_cpy = cur_cpy.right
        return new_op


class LinearOperator(Operator):
    """Basic class for a functional analytic linear operator.
    TODO: write some more
    """
    pass
