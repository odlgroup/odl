# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Entry points for adding more spaces to ODL using external packages.

External packages can add an implementation of `TensorSpace` by hooking
into the setuptools entry point ``'odl.space'`` and exposing the method
``tensor_space_impls``.


Attributes
----------
TENSOR_SPACE_IMPLS: dict
    A dictionary that maps a string to a `TensorSpace` implementation.

Notes
-----
This is used with functions such as `rn`, `cn`, `tensor_space` or
`uniform_discr` in order to allow arbitrary implementations.

See Also
--------
NumpyTensorSpace : Numpy based implementation of `TensorSpace`
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from pkg_resources import iter_entry_points
from odl.space.npy_tensors import NumpyTensorSpace

__all__ = ()


TENSOR_SPACE_IMPLS = {'numpy': NumpyTensorSpace}
for entry_point in iter_entry_points(group='odl.space', name=None):
    try:
        module = entry_point.load()
        TENSOR_SPACE_IMPLS.update(module.tensor_space_impls())
    except ImportError:
        pass
