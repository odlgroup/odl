# Copyright 2014, 2015 The ODL development group
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

"""Discrete wavelet transformation on L2 spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, str

# External
import numpy as np

# Internal
import odl


__all__ = ('DiscreteWaveletTrafo', 'DiscreteWaveletTrafoAdjoint',
           'DiscreteWaveletTrafoInverse')


_SUPPORTED_IMPL = ('pywt',)


def pywt_coeff_to_array(coeff, nscales):
    """Convert a pywt coefficient into a flat array.

    Parameters
    ----------
    coeff : list
        TODO: describe the structure of the list here (shapes, ...).
    nscales : int
        Number of scales in the coefficient list

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        TODO: which shape (length)?

    See also
    --------
    pywt : possibly add a reference to pywavelets here
    """
    # TODO: give the variables better (more telling) names
    # - a -> approx?
    # - dd -> wave_diag?
    # - ...
    a = coeff[0]
    a = a.ravel()
    flat_coeff = a
    for kk in range(1, nscales + 1):
        (dh, dv, dd) = coeff[kk]
        dh = dh.ravel()
        dv = dv.ravel()
        dd = dd.ravel()
        # This seems like slow code - for each scale, a new array is created
        # from the old one by concatenation with the dh, dv and dd
        # Try to determine the array size beforehand, allocate once and
        # fill gradually with, e.g.,  flat_coeff[start:stop] = dh
        flat_coeff = np.concatenate((flat_coeff, dh, dv, dd))
    return flat_coeff


def array_to_pywt_coeff(coef, S, jscale):
    """Convert a flat array into a pywt coefficient list.

    Parameters
    ----------
    TODO: fill in information about the arguments

    Returns
    -------
    :attr:`coeff` : list
        TODO: put in info about the list, basically the same as in
        the description of coeff in pywt_coeff_to_array
    """

    # TODO: give the variables better (more telling) names
    # TODO: check out PEP8 (recommendation: use Spyder and activate the PEP8
    # style checker)
    (n2, m2) = S[0]
    old = n2 * m2
    a = coef[0:old]
    a = np.reshape(a, (n2, m2))
    a = np.asarray(a)
    kk = 1
    W = []
    W.append(a)

    while kk <= jscale:
        (n2, m2) = S[kk]
        maar = n2 * m2
        dh = coef[old:old + maar]
        dh = np.reshape(dh, (n2, m2))
        dh = np.asarray(dh)

        dv = coef[old + maar:old + 2 * maar]
        dv = np.reshape(dv, (n2, m2))
        dv = np.asarray(dv)

        dd = coef[old + 2 * maar:old + 3 * maar]
        dd = np.reshape(dd, (n2, m2))
        dd = np.asarray(dd)

        d = (dh, dv, dd)
        W.append(d)

        kk = kk + 1
        old = old + 3 * maar

    return W


class DiscreteWaveletTrafo(odl.Operator):

    """Discrete wavelet trafo between discrete L2 spaces."""

    # TODO: add arguments to specify wavelet, possibly other options like
    # orthogonal=False etc. (maybe one can ask pywt if a wavelet is
    # orthogonal or biorthogonal?)
    def __init__(self, dom, ran, impl='pywt'):
        """Initialize a new instance.

        Parameters
        ----------
        dom : :class:`~odl.DiscreteLp`
            Domain of the wavelet transform. The exponent ``p`` of the
            discrete :math:`L^p` space must be equal to 2.0.
        ran : :class:`~odl.DiscreteLp`
            Domain of the wavelet transform. The exponent ``p`` of the
            discrete :math:`L^p` space must be equal to 2.0.
        """
        super().__init__(dom, ran)

        if not isinstance(dom, odl.DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))
        if not isinstance(ran, odl.DiscreteLp):
            raise TypeError('range {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))

        if dom.exponent != 2.0:
            raise ValueError('domain Lp exponent is {} instead of 2.0.'
                             ''.format(dom.exponent))
        if ran.exponent != 2.0:
            raise ValueError('range Lp exponent is {} instead of 2.0.'
                             ''.format(ran.exponent))
        if not np.all(dom.grid.stride == 1):
            raise NotImplementedError('non-uniform grid cell sizes not yet '
                                      'supported.')

        self._impl = str(impl).lower()
        if self._impl not in _SUPPORTED_IMPL:
            raise ValueError('implementation {} not supported.'
                             ''.format(impl))

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        # TODO: fill with something. Either from the init method (set a
        # `_is_orthogonal` attribute there and return it here, or get the
        # info from pywt)
        return False

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        # TODO: see is_ortogonal
        return False

    def _apply(self, x, out):
        """Raw in-place application method."""
        # TODO: put the evaluation code here. Maybe pywt allows in-place
        # evaluation, i.e. writing to an existing array?
        # Put the array preparation code into separate functions as you did
        # before

        # `x` is a `DiscreteLp.Vector`. It has a number of convenience
        # methods and attributes
        # - `asarray()`: get the array and reshape
        # - shape
        # ...
        return out

    # TODO: if in-place evaluation is not supported by pywt, implement
    # `_call()` instead of `_apply()`

    @property
    def adjoint(self):
        """The adjoint wavelet transform."""
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return None

    @property
    def inverse(self):
        """The inverse wavelet transform."""
        # TODO: put inverse here
        return None


class DiscreteWaveletTrafoAdjoint(odl.Operator):
    pass


class DiscreteWaveletTrafoInverse(odl.Operator):
    pass
