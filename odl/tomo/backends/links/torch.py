###############################################################################
#          This code was taken from tomosipo and adapted to ODL API           #
#            Please check https://github.com/ahendriksen/tomosipo             #
###############################################################################
"""This module adds support for torch arrays as astra.data3d backends

This module is not automatically imported by tomosipo, you must import
it manually as follows:

>>> import tomosipo as ts
>>> import tomosipo.torch_support

Now, you may use torch tensors as you would numpy arrays:

>>> vg = ts.volume(shape=(10, 10, 10))
>>> pg = ts.parallel(angles=10, shape=10)
>>> A = ts.operator(vg, pg)
>>> x = torch.zeros(A.domain_shape)
>>> A(x).shape == A.range_shape
True

You can also directly apply the tomographic operator to data on the
GPU:

>>> A(x.cuda()).is_cuda
True
"""
import astra
from .base import Link, backends
from .numpy import NumpyLink
from contextlib import contextmanager
import warnings
import torch


class TorchLink(Link):
    """Link implementation for torch arrays"""

    def __init__(self, shape, initial_value):
        super(TorchLink, self).__init__(shape, initial_value)

        if not isinstance(initial_value, torch.Tensor):
            raise ValueError(
                f"Expected initial_value to be a `torch.Tensor'. Got {initial_value.__class__}"
            )

        if initial_value.shape == torch.Size([]):
            self._data = torch.zeros(
                shape, dtype=torch.float32, device=initial_value.device
            )
            self._data[:] = initial_value
        else:
            if shape != initial_value.shape:
                raise ValueError(
                    f"Expected initial_value with shape {shape}. Got {initial_value.shape}"
                )
            # Ensure float32
            if initial_value.dtype != torch.float32:
                warnings.warn(
                    f"The parameter initial_value is of type {initial_value.dtype}; expected `torch.float32`. "
                    f"The type has been automatically converted. "
                    f"Use `ts.link(x.to(dtype=torch.float32))' to inhibit this warning. "
                )
                initial_value = initial_value.to(dtype=torch.float32)
            # Make contiguous:
            if not initial_value.is_contiguous():
                warnings.warn(
                    f"The parameter initial_value should be contiguous. "
                    f"It has been automatically made contiguous. "
                    f"Use `ts.link(x.contiguous())' to inhibit this warning. "
                )
                initial_value = initial_value.contiguous()
            self._data = initial_value

    ###########################################################################
    #                      "Protocol" functions / methods                     #
    ###########################################################################
    @staticmethod
    def __accepts__(initial_value):
        # only accept torch tensors
        return isinstance(initial_value, torch.Tensor)

    def __compatible_with__(self, other):
        dev_self = self._data.device
        if isinstance(other, NumpyLink):
            dev_other = torch.device("cpu")
        elif isinstance(other, TorchLink):
            dev_other = other._data.device
        else:
            return NotImplemented

        return dev_self == dev_other

    ###########################################################################
    #                                Properties                               #
    ###########################################################################
    @property
    def linked_data(self):
        if self._data.is_cuda:
            z, y, x = self._data.shape
            pitch = x * 4  # we assume 4 byte float32 values
            link = astra.data3d.GPULink(self._data.data_ptr(), x, y, z, pitch)
            return link
        else:
            # The torch tensor may be part of the computation
            # graph. It must be detached to obtain a numpy
            # array. We assume that this function will only be
            # called to feed the data into Astra, which should not
            # modify it. So this should be fine.
            return self._data.detach().numpy()

    @property
    def data(self):
        """Returns a shared array with the underlying data.

        Changes to the return value will be reflected in the astra
        data.

        If you want to avoid this, consider copying the data
        immediately, using `x.data.clone()` for instance.

        NOTE: if the underlying object is an Astra projection data
        type, the order of the axes will be in (Y, num_angles, X)
        order.

        :returns: torch.Tensor
        :rtype: torch.Tensor

        """
        return self._data

    @data.setter
    def data(self, val):
        raise AttributeError(
            "You cannot change which torch tensor backs a dataset.\n"
            "To change the underlying data instead, use: \n"
            " >>> vd.data[:] = new_data\n"
        )

    ###########################################################################
    #                             Context manager                             #
    ###########################################################################
    @contextmanager
    def context(self):
        """Context-manager to manage ASTRA interactions

        This context-manager makes sure that the current CUDA
        stream is set to the CUDA device of the current linked data.

        :returns:
        :rtype:

        """
        if self._data.is_cuda:
            with torch.cuda.device_of(self._data):
                yield
        else:
            # no-op for cpu-stored data
            yield

    ###########################################################################
    #                            New data creation                            #
    ###########################################################################
    def new_zeros(self, shape):
        return TorchLink(shape, self._data.new_zeros(shape))

    def new_full(self, shape, value):
        return TorchLink(shape, self._data.new_full(shape, value))

    def new_empty(self, shape):
        return TorchLink(shape, self._data.new_empty(shape))

    def clone(self):
        return TorchLink(self._data.shape, self._data.clone())


# When the torch module is mock imported by the Sphinx documentation system, do
# not alter the observable behavior the linking backend.
if not hasattr(torch, "__sphinx_mock__"):
    backends.append(TorchLink)
