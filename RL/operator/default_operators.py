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
from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import

from future import standard_library

try:
    from builtins import str, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, super

# RL imports
import RL.operator.operator as op
print(op.__file__)

standard_library.install_aliases()


class ScalingOperator(op.SelfAdjointOperator):
    def __init__(self, space, scale):
        self._space = space
        self._scale = scale

    def applyImpl(self, input, out):
        out.linComb(self._scale, input)

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