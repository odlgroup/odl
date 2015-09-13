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


"""ODL is a functional analysis library with a focus on discretization.

ODL suppors abstract sets, linear vector spaces defined on such
and Operators/Functionals defined on these sets. It is intended
to be used to write general code and faciliate code reuse.
"""

from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals

__version__ = '0.1b0.dev0'
__all__ = ('discr', 'operator', 'space')


# Propagate names defined in __all__ of all submodules into the top-level
# module
from . import discr
from .discr.default import *
from .discr.discretization import *
from .discr.grid import *
from .discr.operators import *

__all__ += (discr.default.__all__ + discr.discretization.__all__ +
            discr.grid.__all__ + discr.operators.__all__)

from . import operator
from .operator.default import *
from .operator.operator import *

__all__ += (operator.default.__all__ + operator.operator.__all__)

from . import space
from .space.cartesian import *
from .space.default import *
from .space.domain import *
from .space.function import *
from .space.product import *
from .space.sequence import *
from .space.set import *
from .space.space import *
try:
    from .space.cuda import *
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

__all__ += (space.cartesian.__all__ + space.default.__all__,
            space.domain.__all__, space.function.__all__,
            space.product.__all__, space.sequence.__all__,
            space.space.__all__)
if CUDA_AVAILABLE:
    __all__ += space.cuda.__all__
