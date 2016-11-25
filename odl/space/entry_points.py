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

"""Entry points for adding more spaces to ODL using external packages.

External packages can add implementations of `TensorSet` and `TensorSpace` by
hooking into the setuptools entry point ``'odl.space'`` and exposing the
methods ``tensor_set_impls`` and ``tensor_space_impls``.


Attributes
----------
TENSOR_SET_IMPLS: dict
    A dictionary that maps a string to an `TensorSet` implementation.
TENSOR_SPACE_IMPLS: dict
    A dictionary that maps a string to an `TensorSpace` implementation.

Notes
-----
This is used with functions such as `rn`, `cn`, `tensor_space` or
`uniform_discr` in order to allow arbitrary implementations.

See Also
--------
NumpyTensorSpace : Numpy based implementation of `TensorSpace`
NumpyTensorSet : Numpy based implementation of `TensorSet`
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from pkg_resources import iter_entry_points
from odl.space.npy.tensors import NumpyTensorSet, NumpyTensorSpace

__all__ = ('TENSOR_SET_IMPLS', 'TENSOR_SPACE_IMPLS')

TENSOR_SET_IMPLS = {'numpy': NumpyTensorSet}
TENSOR_SPACE_IMPLS = {'numpy': NumpyTensorSpace}
for entry_point in iter_entry_points(group='odl.space', name=None):
    try:
        module = entry_point.load()
        TENSOR_SET_IMPLS.update(module.tensor_set_impls())
        TENSOR_SPACE_IMPLS.update(module.tensor_space_impls())
    except ImportError:
        pass
