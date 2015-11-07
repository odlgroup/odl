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

"""Concrete vector spaces."""

# TODO: write an introduction

from __future__ import absolute_import

__all__ = ()

from . import ntuples
from .ntuples import *
__all__ += ntuples.__all__

from . import default
from .default import *
__all__ += default.__all__

from . import fspace
from .fspace import *
__all__ += fspace.__all__

try:
    from . import cu_ntuples
    from .cu_ntuples import *
    __all__ += cu_ntuples.__all__
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DTYPES = []

__all__ += ('CUDA_AVAILABLE', 'CUDA_DTYPES')
