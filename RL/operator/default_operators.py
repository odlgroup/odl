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


# Imports for common Python 2/3 codebase
from __future__ import (division, print_function, unicode_literals,
                        absolute_import)

from future import standard_library
from builtins import str, super

# RL imports
import RL.operator.operator as op
from RL.space.space import LinearSpace
from RL.utility.utility import errfmt

standard_library.install_aliases()


class ScalingOperator(op.SelfAdjointOperator):
    def __init__(self, space, scalar):
        if not isinstance(space, LinearSpace):
            raise TypeError(errfmt('''
            'space' ({}) must be a LinearSpace instance
            '''.format(space)))

        self._space = space
        self._scal = float(scalar)

    def _apply(self, input, out):
        out.lincomb(self._scal, input)

    def _call(self, input):
        return self._scal * input

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
    def __init__(self, space):
        super().__init__(space, 1)

    def __repr__(self):
        return 'IdentityOperator(' + repr(self._space) + ')'

    def __str__(self):
        return "I"
