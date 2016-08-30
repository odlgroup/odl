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

"""Discrete wavelet transformation on L2 spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str, super

import numpy as np

from odl.discr import DiscreteLp
from odl.operator import Operator
from odl.trafos.backends.pywt_bindings import (
    pywt_wbasis, pywt_max_level, pywt_wavelet_decomp,
    pywt_wavelet_recon, pywt_coeff_shape_list, PYWT_SUPPORTED_PAD_MODES,
    PYWAVELETS_AVAILABLE)

__all__ = ('WaveletTransform', 'WaveletTransformInverse')


_SUPPORTED_WAVELET_IMPL = ()
if PYWAVELETS_AVAILABLE:
    _SUPPORTED_WAVELET_IMPL += ('pywt',)


class WaveletTransformBase(Operator):

    """Base class for discrete wavelet transforms.

    This abstract class is intended to share code between the forward,
    inverse and adjoint wavelet transforms.
    """

    def __init__(self, space, wbasis, nscales, variant, pad_mode='constant',
                 pad_const=0, impl='pywt'):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp`
            Domain of the forward wavelet transform (the "image domain").
            In the case of ``variant in ('inverse', 'adjoint')``, this
            space is the range of the operator.
        wbasis :  string or `pywt.Wavelet`
            Specification of the wavelet to be used in the transform.
            If a string is given, it is converted to a `pywt.Wavelet`.
            For more information see the `wavelets
            <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_
            page of the PyWavelets documentation.

            Possible wavelet families are:

            'Haar': Haar

            'db': Daubechies

            'sym' : Symlets

            'coif': Coiflets

            'bior': Biorthogonal

            'rbio': Reverse biorthogonal

            'dmey': Discrete FIR approximation of the Meyer wavelet

        nscales : int
            Number of scaling levels to be used in the transform.
            The maximum number of usable scales can be determined
            by `pywt.dwt_max_level`.
        variant : {'forward', 'inverse', 'adjoint'}
            Wavelet transform variant to be created.

        pad_mode : string, optional
            Signal extention mode. Possible extension modes are:

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not doubling the
            outmost values.

            ``'periodic'``: Fill in values from the other side, keeping
            the order.

            ``'order0'``: Extend constantly with the outmost values
            (ensures continuity).

            ``'order1'``: Extend with constant slope (ensures continuity of
            the first derivative). This requires at least 2 values along
            each axis where padding is applied.

            ``'pywt_per'``:  like ``'periodic'``-padding but gives the smallest
            possible number of decomposition coefficients.
            Only available with ``impl='pywt'``, See `pywt.MODES.modes`.

        pad_const : float, optional
            Constant value to use if ``pad_mode == 'constant'``. Ignored
            otherwise.

        impl : {'pywt'}, optional
            Backend for the wavelet transform.

        Examples
        --------
        >>> import odl
        >>> discr_domain = odl.uniform_discr([0, 0], [1, 1], (16, 16))
        >>> op = odl.trafos.WaveletTransform(discr_domain, nscales=1,
        ...                                  wbasis='db1')
        >>> op.is_biorthogonal
        True
        """
        if not isinstance(space, DiscreteLp):
            raise TypeError('`space` {!r} is not a `DiscreteLp` instance.'
                            ''.format(space))
        if space.ndim > 3:
            raise ValueError('`space` can be at most 3-dimensional, got '
                             'ndim={}'.format(space.ndim))

        self.pywt_wbasis = pywt_wbasis(wbasis)

        max_level = pywt_max_level(min(space.shape),
                                   filter_len=self.pywt_wbasis.dec_len)

        nscales, nscales_in = int(nscales), nscales
        if nscales > max_level:
            raise ValueError('cannot use more than {} scaling levels, '
                             'got {}'.format(max_level, nscales_in))
        self.nscales = nscales

        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_WAVELET_IMPL:
            raise ValueError("`impl` '{}' not supported".format(impl_in))
        self.impl = impl

        pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode
        if pad_mode not in PYWT_SUPPORTED_PAD_MODES:
            raise ValueError("`pad_mode` '{}' not supported for `impl` "
                             "'{}'".format(pad_mode_in, self.impl))
        self.pad_mode = pad_mode

        # TODO: move pywt specific check to the backend
        pad_const, pad_const_in = space.field.element(pad_const), pad_const
        if impl == 'pywt' and pad_mode == 'constant' and pad_const != 0:
            raise ValueError("non-zero `pad_const` {} not supported for "
                             "`impl` '{}'".format(pad_const_in, impl))
        self.pad_const = pad_const

        self.shape_list = pywt_coeff_shape_list(
            space.shape, self.pywt_wbasis, self.nscales, self.pad_mode)

        # 1 x approx coeff and (2**n - 1) * detail coeff
        coeff_size = (np.prod(self.shape_list[0]) +
                      sum((2 ** space.ndim - 1) * np.prod(shape)
                          for shape in self.shape_list[1:-1]))

        coeff_space = space.dspace_type(coeff_size, dtype=space.dtype)

        variant, variant_in = str(variant).lower(), variant
        if variant not in ('forward', 'inverse', 'adjoint'):
            raise ValueError("`variant` '{}' not understood"
                             "".format(variant_in))

        if variant == 'forward':
            super().__init__(domain=space, range=coeff_space, linear=True)
        else:
            super().__init__(domain=coeff_space, range=space, linear=True)

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.pywt_wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.pywt_wbasis.biorthogonal


class WaveletTransform(WaveletTransformBase):

    """Discrete wavelet trafo between discretized Lp spaces."""

    def __init__(self, domain, wbasis, nscales, pad_mode='constant',
                 pad_const=0, impl='pywt'):
        """Initialize a new instance.
        Parameters
        ----------
        domain : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
        wbasis :  string or `pywt.Wavelet`
            Specification of the wavelet to be used in the transform.
            If a string is given, it is converted to a `pywt.Wavelet`.
            For more information see the `wavelets
            <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_
            page of the PyWavelets documentation.

            Possible wavelet families are:

            'Haar': Haar

            'db': Daubechies

            'sym' : Symlets

            'coif': Coiflets

            'bior': Biorthogonal

            'rbio': Reverse biorthogonal

            'dmey': Discrete FIR approximation of the Meyer wavelet

        nscales : int
            Number of scaling levels to be used in the transform.
            The maximum number of usable scales can be determined
            by `pywt.dwt_max_level`.
        pad_mode : string, optional
            Signal extention mode. Possible extension modes are:

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not doubling the
            outmost values.

            ``'periodic'``: Fill in values from the other side, keeping
            the order.

            ``'order0'``: Extend constantly with the outmost values
            (ensures continuity).

            ``'order1'``: Extend with constant slope (ensures continuity of
            the first derivative). This requires at least 2 values along
            each axis where padding is applied.

            ``'pywt_per'``:  like ``'periodic'``-padding but gives the smallest
            possible number of decomposition coefficients.
            Only available with ``impl='pywt'``, See `pywt.MODES.modes`.

        pad_const : float, optional
            Constant value to use if ``pad_mode == 'constant'``. Ignored
            otherwise.

        impl : {'pywt'}, optional
            Backend for the wavelet transform.

        Examples
        --------
        >>> import odl
        >>> discr_domain = odl.uniform_discr([0, 0], [1, 1], (16, 16))
        >>> op = WaveletTransform(discr_domain, nscales=1, wbasis='db1')
        >>> op.is_biorthogonal
        True
        """
        super().__init__(domain, wbasis, nscales, 'forward', pad_mode,
                         pad_const, impl)

    def _call(self, x):
        """Compute the discrete wavelet transform.

        Parameters
        ----------
        x : `domain` element

        Returns
        -------
        arr : `numpy.ndarray`
            Flattened and concatenated coefficient array
            The length of the array depends on the size of input image to
            be transformed and on the chosen wavelet basis.
        """
        if self.impl == 'pywt':
            return pywt_wavelet_decomp(x, self.pywt_wbasis, self.pad_mode,
                                       self.nscales, self.shape_list)
        else:
            raise RuntimeError("bad `impl` '{}'".format(self.impl))

    @property
    def adjoint(self):
        """Adjoint wavelet transform.

        Returns
        -------
        adjoint : `WaveletTransformInverse`
            If the transform is orthogonal, the adjoint is the inverse.

        Raises
        ------
        OpNotImplementedError
            if `is_orthogonal` is not ``True``
        """
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return super().adjoint

    @property
    def inverse(self):
        """Inverse wavelet transform.

        Returns
        -------
        inverse : `WaveletTransformInverse`

        See Also
        --------
        adjoint
        """
        return WaveletTransformInverse(
            range=self.domain, nscales=self.nscales, wbasis=self.pywt_wbasis,
            pad_mode=self.pad_mode, pad_const=self.pad_const, impl=self.impl)


class WaveletTransformInverse(WaveletTransformBase):

    """Discrete inverse wavelet trafo between discrete L2 spaces.

    See Also
    --------
    WaveletTransform
    """

    def __init__(self, range, nscales, wbasis, pad_mode='constant',
                 pad_const=0, impl='pywt'):
        """Initialize a new instance.

         Parameters
        ----------
        range : `DiscreteLp`
            Domain of the forward wavelet transform (the "image domain").
        wbasis :  string or `pywt.Wavelet`
            Specification of the wavelet to be used in the transform.
            If a string is given, it is converted to a `pywt.Wavelet`.
            For more information see the `wavelets
            <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_
            page of the PyWavelets documentation.

            Possible wavelet families are:

            'Haar': Haar

            'db': Daubechies

            'sym' : Symlets

            'coif': Coiflets

            'bior': Biorthogonal

            'rbio': Reverse biorthogonal

            'dmey': Discrete FIR approximation of the Meyer wavelet

        nscales : int
            Number of scaling levels to be used in the transform.
            The maximum number of usable scales can be determined
            by `pywt.dwt_max_level`.
        pad_mode : string, optional
            Signal extention mode. Possible extension modes are:

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not doubling the
            outmost values.

            ``'periodic'``: Fill in values from the other side, keeping
            the order.

            ``'order0'``: Extend constantly with the outmost values
            (ensures continuity).

            ``'order1'``: Extend with constant slope (ensures continuity of
            the first derivative). This requires at least 2 values along
            each axis where padding is applied.

            ``'pywt_per'``:  like ``'periodic'``-padding but gives the smallest
            possible number of decomposition coefficients.
            Only available with ``impl='pywt'``, See `pywt.MODES.modes`.

        pad_const : float, optional
            Constant value to use if ``pad_mode == 'constant'``. Ignored
            otherwise.

        impl : {'pywt'}, optional
            Backend for the wavelet transform.
        """
        super().__init__(range, wbasis, nscales, 'inverse', pad_mode,
                         pad_const, impl)

    def _call(self, coeff):
        """Compute the discrete 1D, 2D or 3D inverse wavelet transform.

        Parameters
        ----------
        coeff : `domain` element
            Wavelet coefficients supplied to the wavelet reconstruction.

        Returns
        -------
        arr : `numpy.ndarray`
            Result of the wavelet reconstruction.
        """
        if self.impl == 'pywt':
            return pywt_wavelet_recon(coeff, self.pywt_wbasis, self.pad_mode,
                                      self.shape_list)
        else:
            raise RuntimeError("bad `impl` '{}'".format(self.impl))

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `WaveletTransform`
            If the transform is orthogonal, the adjoint is the inverse.

        Raises
        ------
        OpNotImplementedError
            if `is_orthogonal` is not ``True``

        See Also
        --------
        inverse
        """
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return super().adjoint

    @property
    def inverse(self):
        """Inverse of this operator.

        Returns
        -------
        inverse : `WaveletTransform`

        See Also
        --------
        adjoint
        """
        return WaveletTransform(
            domain=self.range, nscales=self.nscales, wbasis=self.pywt_wbasis,
            pad_mode=self.pad_mode, pad_const=self.pad_const, impl=self.impl)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
