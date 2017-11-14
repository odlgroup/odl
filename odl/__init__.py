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
__all__ = ('set',
           'space',
           'operator',
           'discr',
           'contrib',
           'deform',
           'diagnostics',
           'phantom',
           'solvers',
           'tomo',
           'trafos',
           'ufunc_ops',
           'util',
           )

# Propagate names defined in` __all__` of all "core" subpackages into
# the top-level namespace
from .set import *
__all__ += set.__all__

from .space import *
__all__ += space.__all__

from .operator import *
__all__ += operator.__all__

from .discr import *
__all__ += discr.__all__

# More "advanced" subpackages keep their namespaces separate from top-level,
# we only import the modules themselves
from . import contrib
from . import deform
from . import diagnostics
from . import phantom
from . import solvers
from . import tomo
from . import trafos
from . import ufunc_ops
from . import util

# Add `test` function to global namespace so users can run `odl.test()`
from .util import test
__all__ += ('test',)
