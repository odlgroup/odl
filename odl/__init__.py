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


"""ODL is a functional analysis library with a focus on discretization.

ODL suppors abstract sets, linear vector spaces defined on such
and Operators/Functionals defined on these sets. It is intended
to be used to write general code and faciliate code reuse.
"""

from __future__ import absolute_import

__version__ = '0.5.1'
__all__ = ('diagnostics', 'discr', 'operator', 'set', 'space', 'solvers',
           'tomo', 'trafos', 'util', 'phantom', 'deform')

# Propagate names defined in __all__ of all submodules into the top-level
# module
from . import diagnostics

from .discr import *
__all__ += discr.__all__

from .operator import *
__all__ += operator.__all__

from .set import *
__all__ += set.__all__

from .space import *
__all__ += space.__all__

from . import solvers
from . import trafos
from . import tomo
from . import util
from . import phantom
from . import deform

from .util import test
__all__ += ('test',)
