# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Entry points for adding more spaces to ODL using external packages.

External packages can add an implementation of `TensorSpace` by hooking
into the setuptools entry point ``'odl.space'`` and exposing the methods
``tensor_space_impl`` and ``tensor_space_impl_names``.

This is used with functions such as `rn`, `cn`, `tensor_space` or
`uniform_discr` in order to allow arbitrary implementations.

See Also
--------
NumpyTensorSpace : Numpy-based implementation of `TensorSpace`
"""

from __future__ import print_function, division, absolute_import

from odl.space.npy_tensors import NumpyTensorSpace

# We don't expose anything to odl.space
__all__ = ()

IS_INITIALIZED = False
TENSOR_SPACE_IMPLS = {'numpy': NumpyTensorSpace}


def _initialize_if_needed():
    """Initialize ``TENSOR_SPACE_IMPLS`` if not already done."""
    global IS_INITIALIZED, TENSOR_SPACE_IMPLS
    if not IS_INITIALIZED:
        # pkg_resources has long import time
        from pkg_resources import iter_entry_points
        for entry_point in iter_entry_points(group='odl.space', name=None):
            try:
                module = entry_point.load()
            except ImportError:
                pass
            else:
                TENSOR_SPACE_IMPLS.update(module.tensor_space_impls())
        IS_INITIALIZED = True


def tensor_space_impl_names():
    """A tuple of strings with valid tensor space implementation names."""
    _initialize_if_needed()
    return tuple(TENSOR_SPACE_IMPLS.keys())


def tensor_space_impl(impl):
    """Tensor space class corresponding to the given impl name.

    Parameters
    ----------
    impl : str
        Name of the implementation, see `tensor_space_impl_names` for
        the full list.

    Returns
    -------
    tensor_space_impl : type
        Class inheriting from `TensorSpace`.

    Raises
    ------
    ValueError
        If ``impl`` is not a valid name of a tensor space imlementation.
    """
    if impl != 'numpy':
        # Shortcut to improve "import odl" times since most users do not use
        # non-numpy backends
        _initialize_if_needed()

    try:
        return TENSOR_SPACE_IMPLS[impl]
    except KeyError:
        raise ValueError("`impl` {!r} does not correspond to a valid tensor "
                         "space implmentation".format(impl))
