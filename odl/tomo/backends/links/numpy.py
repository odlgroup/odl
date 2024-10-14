###############################################################################
#          This code was taken from tomosipo and adapted to ODL API           #
#            Please check https://github.com/ahendriksen/tomosipo             #
###############################################################################

import numpy as np
import warnings
from contextlib import contextmanager
from .base import Link, backends


class NumpyLink(Link):
    """Link implementation for numpy arrays"""

    def __init__(self, shape, initial_value):
        super(NumpyLink, self).__init__(shape, initial_value)

        if initial_value is None:
            self._data = np.zeros(shape, dtype=np.float32)
        elif np.isscalar(initial_value):
            self._data = np.zeros(shape, dtype=np.float32)
            self._data[:] = initial_value
        else:
            initial_value = np.array(initial_value, copy=False)
            if initial_value.shape != shape:
                raise ValueError(
                    "Cannot link array. "
                    f"Expected array of shape {shape}. Got {initial_value.shape}"
                )
            # Make contiguous:
            if initial_value.dtype != np.float32:
                warnings.warn(
                    f"The parameter initial_value is of type {initial_value.dtype}; expected `np.float32`. "
                    f"The type has been Automatically converted. "
                    f"Use `ts.link(x.astype(np.float32))' to inhibit this warning. "
                )
                initial_value = initial_value.astype(np.float32)
            if not (
                initial_value.flags["C_CONTIGUOUS"] and initial_value.flags["ALIGNED"]
            ):
                warnings.warn(
                    f"The parameter initial_value should be C_CONTIGUOUS and ALIGNED. "
                    f"It has been automatically made contiguous and aligned. "
                    f"Use `ts.link(np.ascontiguousarray(x))' to inhibit this warning. "
                )
                initial_value = np.ascontiguousarray(initial_value)
            self._data = initial_value

    ###########################################################################
    #                      "Protocol" functions / methods                     #
    ###########################################################################
    @staticmethod
    def __accepts__(initial_value):
        # `NumpyLink' is the default backend, so it should accept
        # an initial_value of `None'.
        if initial_value is None:
            return True
        elif isinstance(initial_value, np.ndarray):
            return True
        elif np.isscalar(initial_value):
            return True
        else:
            return False

    def __compatible_with__(self, other):
        if isinstance(other, NumpyLink):
            return True
        else:
            return NotImplemented

    ###########################################################################
    #                                Properties                               #
    ###########################################################################
    @property
    def linked_data(self):
        """Returns a numpy array or GPULink object

        :returns:
        :rtype:

        """
        return self._data

    @property
    def data(self):
        """Returns a shared numpy array with the underlying data.

        Changes to the return value will be reflected in the astra
        data.

        If you want to avoid this, consider copying the data
        immediately, using `np.copy` for instance.

        NOTE: if the underlying object is an Astra projection data
        type, the order of the axes will be in (Y, num_angles, X)
        order.

        :returns: np.array
        :rtype: np.array

        """
        return self._data

    @data.setter
    def data(self, val):
        raise AttributeError(
            "You cannot change which array backs a dataset.\n"
            "To change the underlying data instead, use: \n"
            " >>> x.data[:] = new_data\n"
        )

    ###########################################################################
    #                             Context manager                             #
    ###########################################################################
    @contextmanager
    def context(self):
        """Context-manager to manage ASTRA interactions

        This is a no-op for numpy data.

        """
        yield

    ###########################################################################
    #                            New data creation                            #
    ###########################################################################
    def new_zeros(self, shape):
        return NumpyLink(
            shape,
            np.zeros(shape, dtype=self._data.dtype),
        )

    def new_full(self, shape, value):
        return NumpyLink(
            shape,
            np.full(shape, value, dtype=self._data.dtype),
        )

    def new_empty(self, shape):
        return NumpyLink(
            shape,
            np.empty(shape, dtype=self._data.dtype),
        )

    def clone(self):
        return NumpyLink(self._data.shape, np.copy(self._data))


backends.append(NumpyLink)
