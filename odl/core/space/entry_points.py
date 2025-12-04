# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Entry points for adding more spaces to ODL using external packages.

External packages can add an implementation of `TensorSpace` by hooking
into the setuptools entry point ``'odl.core.space'`` and exposing the methods
``tensor_space_impl`` and ``tensor_space_impl_names``.

This is used with functions such as `rn`, `cn`, `tensor_space` or
`uniform_discr` in order to allow arbitrary implementations.

See Also
--------
NumpyTensorSpace : Numpy-based implementation of `TensorSpace`
"""

# pylint: disable=line-too-long
# We want to import if the backends are actually available
# pylint: disable=import-outside-toplevel
# We want to use a global statement here
# pylint: disable=global-statement
# The global variable TENSOR_SPACE_IMPLS is modified in a condition, which triggers the pylint warning
# pylint: disable=global-variable-not-assigned

from odl.backends.arrays.npy_tensors import NumpyTensorSpace

# We don't expose anything to odl.core.space
__all__ = ()

is_initialized = False
TENSOR_SPACE_IMPLS = {"numpy": NumpyTensorSpace}


def _initialize_if_needed():
    """Initialize ``TENSOR_SPACE_IMPLS`` if not already done."""
    global is_initialized, TENSOR_SPACE_IMPLS
    if not is_initialized:
        import importlib.util

        torch_module = importlib.util.find_spec("torch")
        if torch_module is not None:
            try:
                from odl.backends.arrays.pytorch_tensors import PyTorchTensorSpace

                TENSOR_SPACE_IMPLS["pytorch"] = PyTorchTensorSpace
            except ModuleNotFoundError:
                pass
        is_initialized = True


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
    try:
        return TENSOR_SPACE_IMPLS[impl]
    except KeyError:
        raise KeyError(
            f"`impl` {impl} does not correspond to a valid tensor "
            "space implmentation"
        )


_initialize_if_needed()
