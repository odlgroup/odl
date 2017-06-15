# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""ODL (Operator Discretization Library).

ODL is a Python library for fast prototyping focusing on (but not
restricted to) inverse problems.
"""

from __future__ import absolute_import

__version__ = '0.6.1.dev0'
__all__ = ('diagnostics', 'discr', 'operator', 'set', 'space', 'solvers',
           'tomo', 'trafos', 'util', 'phantom', 'deform', 'ufunc_ops',
           'datasets', 'contrib')

# Propagate names defined in __all__ of all submodules into the top-level
# module
from .set import *
__all__ += set.__all__

# operator must come before space because npy_ntuples imports Operator
from .operator import *
__all__ += operator.__all__

from .space import *
__all__ += space.__all__

from .discr import *
__all__ += discr.__all__

from . import diagnostics
from . import solvers
from . import trafos
from . import tomo
from . import util
from . import phantom
from . import deform
from . import ufunc_ops
from . import datasets
from . import contrib


from .util import test
__all__ += ('test',)
