# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Entry points for adding more spaces to ODL using external packages.

External packages can add implementations of `NtuplesBase` and `FnBase` by
hooking into the setuptools entry point ``'odl.space'`` and exposing the
methods ``ntuples_impls`` and ``fn_impls``.


Attributes
----------
NTUPLES_IMPLS: dict
    A dictionary that maps a string to an `NtuplesBase` implementation.
FN_IMPLS: dict
    A dictionary that maps a string to an `FnBase` implementation.

Notes
-----
This is used with functions such as `rn`, `fn` and `uniform_discr` in order
to allow arbitrary implementations.

See Also
--------
NumpyFn : Numpy based implementation of `FnBase`
NumpyNtuples : Numpy based implementation of `NtuplesBase`
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from pkg_resources import iter_entry_points
from odl.space.npy_ntuples import NumpyNtuples, NumpyFn

__all__ = ('NTUPLES_IMPLS', 'FN_IMPLS')

NTUPLES_IMPLS = {'numpy': NumpyNtuples}
FN_IMPLS = {'numpy': NumpyFn}
for entry_point in iter_entry_points(group='odl.space', name=None):
    try:
        module = entry_point.load()
        NTUPLES_IMPLS.update(module.ntuples_impls())
        FN_IMPLS.update(module.fn_impls())
    except ImportError:
        pass
