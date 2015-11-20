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

"""Discrete wavelet transformation on :math:`L^p` spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, str, super

# External
from math import pi
import numpy as np
try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    pyfftw = None
    PYFFTW_AVAILABLE = False

# Internal
from odl.discr.grid import RegularGrid
from odl.discr.lp_discr import DiscreteLp
from odl.operator.operator import Operator
from odl.set.sets import ComplexNumbers


__all__ = ('DiscreteFourierTrafo', 'DiscreteFourierTrafoInverse')


_SUPPORTED_IMPL = ('pyfftw',)


def reciprocal(grid, halfcomplex=False, even_shift=False):
    """Return the reciprocal of the given regular grid.

    This function calculates the reciprocal (Fourier/frequency space)
    grid for a given regular grid defined by the nodes

        :math:`x_k = x_0 + k \odot s, \quad k \\in I_N`

    with an index set :math:`I_N = \{k\\in\mathbb{Z}\ |\ 0\\leq k< N\}`.
    The multiplication ":math:`\odot`" and the comarisons are to be
    understood component-wise.

    This grid's reciprocal is then given by the nodes

        :math:`\\xi_j = \\xi_0 + j \odot \sigma, \quad
        \sigma = 2\pi (s \odot N)^{-1}`.

    The minimum frequency :math:`\\xi_0` can in principle be chosen
    freely, but usually it is chosen in a such a way that the reciprocal
    grid is centered around zero. For this, there are two possibilities:

    1. Make the grid point-symmetric around 0.

    2. Make the grid "almost" point-symmetric around zero but guarantee
       that 0 is a grid point.

    For an odd number :math:`n` of samples (in one of the dimensions),
    both options are equivalent, and the minimum frequency is given as

        :math:`\\xi_{0, \\text{odd}} = -\pi s + \pi / (sn) =
        -\pi s + \sigma/2`.

    For :math:`n` even, the first option results in the same formula as
    in the odd case, and the second option can be realized by shifting
    by half a cell size either to the left:

        :math:`\\xi_{0, \\text{even, left}} = -\pi s =
        \\xi_{0, \\text{odd}} - \sigma/2.`

    Parameters
    ----------
    grid : :class:`~odl.RegularGrid`
        Original sampling grid
    halfcomplex : `bool`, optional
        If `True`, return the half of the grid with last coordinate
        less than zero. This is related to the fact that for real-valued
        functions, the other half is the mirrored complex conjugate of
        the given half and therefore needs not be stored.
    even_shift : `bool` or iterable, optional
        In dimensions with even number of samples, the point-symmetric
        reciprocal grid does not contain 0. If `True`, the grid is
        shifted by half a stride in the negative direction
        such that 0 falls on a grid point.
        With a boolean array or iterable, this option is applied
        separately on each axis. At least ``grid.ndim`` values must be
        provided.

    Returns
    -------
    recip : :class:`~odl.RegularGrid`
        The reciprocal grid
    """
    try:
        even_shift = [bool(even_shift[i]) for i in range(grid.ndim)]
    except IndexError:
        raise ValueError('boolean shift iterable gives too few entries '
                         '({} < {}).'.format(i, grid.ndim))
    except TypeError:
        even_shift = [bool(even_shift)] * grid.ndim

    rmin = np.empty_like(grid.min_pt)
    rmax = np.empty_like(grid.max_pt)
    rsamples = list(grid.shape)

    stride = grid.stride
    shape = np.array(grid.shape)

    # Odd axes -> simply make symmetric
    odd = np.where(shape % 2 == 1)
    rmin[odd] = (-1.0 + 1.0 / shape[odd]) * pi / stride[odd]
    rmax[odd] = -rmin[odd]

    # Even shifted axes
    even_s = np.where(np.logical_and(shape % 2 == 0, even_shift))
    rmin[even_s] = -pi / stride[even_s]
    # Length min->max increases by double the shift, so we
    # have to compensate by a full stride
    rmax[even_s] = (-rmin[even_s] - 2 * pi / (stride[even_s] * shape[even_s]))

    # Even non-shifted axes
    even_no_s = np.where(np.logical_and(shape % 2 == 0,
                                        np.logical_not(even_shift)))
    rmin[even_no_s] = (-1.0 + 1.0 / shape[even_no_s]) * pi / stride[even_no_s]
    rmax[even_no_s] = -rmin[even_no_s]

    # Change last axis shape and max if halfcomplex
    if halfcomplex:
        rsamples[-1] = shape[-1] // 2 + 1
        if shape[-1] % 2 == 1 or even_shift[-1]:
            rmax[-1] = 0
        else:
            rmax[-1] = pi / (shape[-1] * stride[-1])

    return RegularGrid(rmin, rmax, rsamples, as_midp=False)


class DiscreteFourierTrafo(Operator):

    """Discrete Fourier trafo between discrete Lp spaces.

    The Fourier transform is defined as the linear operator

        :math:`\mathcal{F}: L^p(\mathbb{R}^d) \\to L^q(\mathbb{R}^d)`,

        :math:`\mathcal{F}(\phi)(\\xi) = \widehat{\phi}(\\xi) =
        (2\pi)^{-\\frac{d}{2}}
        \int_{\mathbb{R}^d} \phi(x)\ e^{-i x^{\mathrm{T}}\\xi}
        \ \mathrm{d}x`,

    where :math:`p \geq 1` and :math:`q = p / (p-1)`. The Fourier
    transform is bounded for :math:`1 \\leq p \\leq 2` and can be
    reasonably defined for :math:`p > 2` in the distributional sense
    [1]_.
    Its inverse is given by the formula

        :math:`\mathcal{F^{-1}}(\phi)(x) = \widetilde{\phi}(\\xi) =
        (2\pi)^{-\\frac{d}{2}}
        \int_{\mathbb{R}^d} \phi(\\xi)\ e^{i \\xi^{\mathrm{T}}x}
        \ \mathrm{d}\\xi`,

    For :math:`p = 2`, it is :math:`q = 2`, and the inverse Fourier
    transform is the adjoint operator,
    :math:`\mathcal{F}^* = \mathcal{F}^{-1}`. Note that

        :math:`\mathcal{F^{-1}}(\phi) = \mathcal{F}(\check\phi)
        = \mathcal{F}(\phi)(-\cdot)
        = \overline{\mathcal{F}(\overline{\phi})}
        = \mathcal{F}^3(\phi), \quad \check\phi(x) = \phi(-x)`.

    This implies in particular that for real-valued :math:`\phi`,
    it is :math:`\overline{\mathcal{F}(\phi)}(\\xi) =
    \mathcal{F}(\phi)(-\\xi)`, i.e. the Fourier transform is completely
    known already from the its values in a half-space only. This
    property is used in the `halfcomplex storage format
    <http://fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html>`_.

    Further properties are summarized in `the Wikipedia article on
    the Fourier transform
    <https://en.wikipedia.org/wiki/Fourier_transform>`_.

    References
    ----------
    .. [1] Stein, Elias and Weiss, Guido (1971). Introduction to
       Fourier Analysis on Euclidean Spaces. Princeton, N.J.:
       Princeton University Press. ISBN 978-0-691-08078-9

    """

    def __init__(self, dom, ran=None, **kwargs):
        """
        Parameters
        ----------
        dom : :class:`~odl.DiscreteLp`
            Domain of the wavelet transform. Its
            :attr:`~odl.DiscreteLp.exponent` must be at least 1.0;
            if it is equal to 2.0, this operator has an adjoint which
            is equal to the inverse.

        ran : :class:`~odl.DiscreteLp`, optional
            Domain of the wavelet transform. The exponent :math:`q` of
            the discrete :math:`L^q` space must be equal to
            :math:`p/(p-1)`, where :math:`p` is the exponent of ``dom``.
            Note that for :math:`p=2`, it is :math:`q=2`.

            If ``ran`` is `None`, the :attr:`~odl.DiscreteLp.grid`
            of ``dom`` must be a :class:`odl.RegularGrid`. In this case,
            the operator range is created as :class:`~odl.DiscreteLp`
            with conjugate exponent :math:`p/(p-1)` and reciprocal grid
            sampling.

        kwargs : {'halfcomplex', 'even_shift'}

            'halfcomplex' : `bool`, optional
                If `True`, calculate only the negative frequency part
                for real input. If `False`, calculate the full
                complex FFT.

                This option only applies to 'uni-to-uni' transforms.
                For complex domain, it has no effect.

                Default: `False`

            'even_shift' : {'none', 'left', 'right'}
                In dimensions with even number of samples, the
                point-symmetric reciprocal grid does not contain 0.
                For options ``'left'`` and ``'right'``, the grid is
                shifted by half a stride in the specified direction
                such that 0 falls on a grid point.

                Inputs other than ``'left'`` or ``'right'`` are
                interpreted as ``'none'``.

                This option only applies to 'uni-to-uni' transforms.

                Default: ``'none'``

        Notes
        -----
        The :attr:`~odl.Operator.range` of this operator always has the
        :class:`~odl.ComplexNumbers` as its
        :attr:`~odl.LinearSpace.field`, i.e. if the field of ``dom``
        is the :class:`~odl.RealNumbers`, this operator has no
        :attr:`~odl.Operator.adjoint`.
        """
        if not isinstance(dom, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))

        # Check exponents
        if dom.exponent < 1:
            raise ValueError('domain exponent {} < 1 not allowed.'
                             ''.format(dom.exponent))
        if dom.exponent == 1.0:
            conj_exp = float('inf')
        else:
            conj_exp = dom.exponent / (dom.exponent - 1.0)

        if ran is not None:
            if not isinstance(ran, DiscreteLp):
                raise TypeError('range {!r} is not a `DiscreteLp` instance.'
                                ''.format(dom))
            if not np.allclose(ran.exponent, conj_exp):
                raise ValueError('range exponent {} not equal to the '
                                 'conjugate exponent {}.'
                                 ''.format(self.range.exponent, conj_exp))

        if not isinstance(dom.grid, RegularGrid):
            raise NotImplementedError('irregular grids not supported yet.')
        else:
            if dom.field == ComplexNumbers():
                self._halfcomplex = False
            else:
                self._halfcomplex = bool(kwargs.pop('halfcomplex'), False)

            even_shift = str(kwargs.pop('even_shift')).lower()
            if even_shift not in ('left', 'right'):
                even_shift = 'none'

            recip_grid = reciprocal(dom.grid, halfcomplex=self._halfcomplex,
                                    even_shift=even_shift)

            pass

        super().__init__(dom, ran, linear=True)

        # TODO: relax this restraint
        if not np.all(dom.grid.stride == 1):
            raise NotImplementedError('non-uniform grid cell sizes not yet '
                                      'supported.')

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
        # TODO: add appropriate arguments
        return DiscreteFourierTrafoInverse()


class DiscreteFourierTrafoInverse(Operator):
    pass


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
