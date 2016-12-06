# Copyright 2014-2016 The ODL development group
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

"""Basic vector spaces and utilities."""

from __future__ import absolute_import

__all__ = ('base_tensors', 'weighting')

from . import base_tensors
from . import weighting

from .npy_tensors import *
__all__ += npy_tensors.__all__

from .pspace import *
__all__ += pspace.__all__

from .fspace import *
__all__ += fspace.__all__

from .entry_points import *
__all__ += entry_points.__all__

from .space_utils import *
__all__ += space_utils.__all__
