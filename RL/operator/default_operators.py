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


from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()

import RL.operator.operator as op


class IdentityOperator(op.SelfAdjointOperator):
    def __init__(self, space):
        self._space = space

    def applyImpl(self, input, out):
        out.assign(input)

    @property
    def domain(self):
        return self._space

    @property
    def range(self):
        return self._space

    def __repr__(self):
        return "IdentityOperator(" + repr(self._space) + ")"

    def __str__(self):
        return "I"