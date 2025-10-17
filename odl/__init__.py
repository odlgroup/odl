# Copyright 2014-2020 The ODL contributors
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

from os import pardir, path, environ
environ['SCIPY_ARRAY_API']='1'

import numpy as np

__all__ = (
    'array_API_support'
    'set',
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
)

# Set package version
curdir = path.abspath(path.dirname(__file__))

with open(path.join(curdir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

# Set old Numpy printing behavior as to not invalidate all doctests.
# TODO(kohr-h): switch to new behavior when Numpy 1.14 is minimum
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass

# Set printing line width to 71 to allow method docstrings to not extend
# beyond 79 characters (2 times indent of 4)
np.set_printoptions(linewidth=71)

# Import all names from "core" subpackages into the top-level namespace;
# the `__all__` collection is extended later to make import errors more
# visible (otherwise one gets errors like "... has no attribute __all__")
from .core.set import *
from .core.array_API_support import *
from .core.discr import *
from .core.operator import *
from .core.space import *

# More "advanced" subpackages keep their namespaces separate from top-level,
# we only import the modules themselves
from . import contrib
from . import deform
from . import diagnostics
from .core import phantom
from . import solvers
from . import tomo
from . import trafos
# from . import ufunc_ops

# Add `test` function to global namespace so users can run `odl.test()`
from .core.util import test


# Amend `__all__`
__all__ += ('test',)
