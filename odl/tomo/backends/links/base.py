###############################################################################
#          This code was taken from tomosipo and adapted to ODL API           #
#            Please check https://github.com/ahendriksen/tomosipo             #
###############################################################################

from contextlib import contextmanager
import warnings

backends = [
    # numpy backend is imported by default;
    # torch backend is only supported when the PyTorch package has been installed;
    # cupy backend is only supported when the CuPy package has been installed.
]

def link(arr, shape):
    for backend in backends:
        if backend.__accepts__(arr):
            return backend(shape, arr)
    raise ValueError(f"An initial_value of class {type(arr)} is not supported. ")

def are_compatible(link_a, link_b):
    a_compat_with_b = link_a.__compatible_with__(link_b)
    if a_compat_with_b is True:
        return True
    elif a_compat_with_b == NotImplemented:
        b_compat_with_a = link_b.__compatible_with__(link_a)
        if b_compat_with_a is True:
            return True
        elif b_compat_with_a == NotImplemented:
            warnings.warn(
                f"Cannot determine if link of type {type(link_a)} is compatible with {type(link_b)}. "
                "Continuing anyway."
            )
        else:
            return False
    else:
        return False


class Link(object):
    """A General base class for link types"""

    def __init__(self, shape, initial_value):
        self._shape = shape
        super().__init__()

    ###########################################################################
    #                      "Protocol" functions / methods                     #
    ###########################################################################
    @staticmethod
    def __accepts__(initial_value):
        """Determines if the link class can make use of the initial_value

        :param initial_value:
        :returns:
        :rtype:

        """
        raise NotImplementedError()

    def __compatible_with__(self, other):
        """Can ASTRA project from this link to other link?"""
        raise NotImplementedError()

    ###########################################################################
    #                                Properties                               #
    ###########################################################################
    @property
    def linked_data(self):
        """Returns a numpy array or GPULink object

        :returns:
        :rtype:

        """
        raise NotImplementedError()

    @property
    def data(self):
        """Returns the underlying data.

        Changes to the return value will be reflected in the astra
        data.
        """
        raise NotImplementedError()

    @data.setter
    def data(self, val):
        raise AttributeError(
            "You cannot change which array backs a dataset.\n"
            "To change the underlying data instead, use: \n"
            " >>> x.data[:] = new_data\n"
        )

    @property
    def shape(self):
        return self._shape

    ###########################################################################
    #                             Context manager                             #
    ###########################################################################
    @contextmanager
    def context(self):
        """Context-manager to manage ASTRA interactions

        This is a no-op for numpy data.

        This context-manager used, for example, for pytorch data on
        GPU to make sure the current CUDA stream is set to the device
        of the input data.

        :returns:
        :rtype:

        """
        raise NotImplementedError()

    ###########################################################################
    #                            New data creation                            #
    ###########################################################################
    def new_zeros(self, shape):
        raise NotImplementedError()

    def new_full(self, shape, value):
        raise NotImplementedError()

    def new_empty(self, shape):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()
