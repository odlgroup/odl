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


"""
RL is a functional analysis library

RL suppors abstract sets, linear vector spaces defined on such
and Operators/Functionals defined on these sets. It is intended
to be used to write general code and faciliate code reuse.
"""

from __future__ import absolute_import

__version__ = '0.1b0.dev0'

__all__ = ['geometry', 'operator', 'space', 'utility']

import RL.geometry
import RL.operator
import RL.space
import RL.utility
