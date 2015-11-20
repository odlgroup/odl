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
try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    pyfftw = None
    PYFFTW_AVAILABLE = False

# Internal
import odl
from odl.util.utility import almost_equal


__all__ = ('DiscreteFourierTrafo', 'DiscreteFourierTrafoInverse')


_SUPPORTED_IMPL = ('pyfftw',)


class DiscreteFourierTrafo(odl.Operator):

    """Discrete Fourier trafo between discrete Lp spaces.

    The Fourier transform is defined as the bounded linear operator

        :math:`\mathcal{F}: L^p(\mathbb{R}^d) \\to L^q(\mathbb{R}^d)`,
        :math:`\mathcal{F}(\phi)(\xi) = \widehat{\phi}(\xi) =
        (2\pi)^{-\\frac{d}{2}}
        \int_{\mathbb{R}^d} \phi(x)\ e^{-i x^{\mathrm{T}}\xi}
        \ \mathrm{d}x`,

    where :math:`1 \\leq p \\leq 2` and :math:`q = p / (p-1)`. Its
    inverse is given by the formula

        :math:`\mathcal{F^{-1}}(\phi)(x) = \widecheck{\phi}(\xi) =
        (2\pi)^{-\\frac{d}{2}}
        \int_{\mathbb{R}^d} \phi(\xi)\ e^{i \xi^{\mathrm{T}}x}
        \ \mathrm{d}\xi`,

    For
    :math:`p = 2`, it is :math:`q = 2`, and the Fourier transform
    """

    def __init__(self, ran=None, impl='pyfftw'):
        """Initialize a new instance.

        Parameters
        ----------
        dom : :class:`~odl.DiscreteLp`
            Domain of the wavelet transform. If the
            :attr:`~odl.DiscreteLp.exponent` :math:`p` of
            the discrete :math:`L^p` is 2.0, it has an adjoint which
            is equal to its inverse.
        ran : :class:`~odl.DiscreteLp`, optional
            Domain of the wavelet transform. The exponent :math:`q` of
            the discrete :math:`L^q` space must be equal to
            :math:`p/(p-1)`, where :math:`p` is the exponent of ``dom``.
            Note that for :math:`p=2`, it is :math:`q=2`.
        """
        if not isinstance(dom, odl.DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))
        if not isinstance(ran, odl.DiscreteLp):
            raise TypeError('range {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))

        conj_exp = dom.exponent / (dom.exponent - 1.0)
        if not almost_equal(ran.exponent, conj_exp):
            raise ValueError('Range exponent {} not equal to the conjugate '
                             'exponent {}.'.format(self.range.exponent,
                                                   conj_exp))

        super().__init__(dom, ran)

        if not np.all(dom.grid.stride == 1):
            raise NotImplementedError('non-uniform grid cell sizes not yet '
                                      'supported.')

        self._impl = str(impl).lower()
        if self._impl not in _SUPPORTED_IMPL:
            raise ValueError('implementation {} not supported.'
                             ''.format(impl))

    def _apply(self, x, out):
        """Raw in-place application method."""
        pass

    @property
    def adjoint(self):
        """The adjoint wavelet transform."""
        if self.domain.exponent == 2.0:
            return self.inverse
        else:
            raise NotImplementedError('adjoint only defined for exponent 2.0, '
                                      'not {}.'.format(self.domain.exponent))

    @property
    def inverse(self):
        """The inverse wavelet transform."""
        # TODO: put inverse here
        return None


#class DiscreteFourierTrafoInverse(odl.Operator):
#    pass


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
