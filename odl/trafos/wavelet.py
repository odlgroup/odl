# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Discrete wavelet transformation on L2 spaces."""

from __future__ import print_function, division, absolute_import

import numpy as np
from odl.discr import DiscreteLp
from odl.operator import Operator
from odl.trafos.backends.pywt_bindings import (
    PYWT_AVAILABLE,
    pywt_pad_mode, pywt_wavelet, precompute_raveled_slices)

__all__ = ('WaveletTransform', 'WaveletTransformInverse')


_SUPPORTED_WAVELET_IMPLS = ()
if PYWT_AVAILABLE:
    _SUPPORTED_WAVELET_IMPLS += ('pywt',)
    import pywt


class WaveletTransformBase(Operator):

    """Base class for discrete wavelet transforms.

    This abstract class is intended to share code between the forward,
    inverse and adjoint wavelet transforms.
    """

    def __init__(self, space, wavelet, nlevels, variant, pad_mode='constant',
                 pad_const=0, impl='pywt', axes=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp`
            Domain of the forward wavelet transform (the "image domain").
            In the case of ``variant in ('inverse', 'adjoint')``, this
            space is the range of the operator.
        wavelet : string or `pywt.Wavelet`
            Specification of the wavelet to be used in the transform.
            If a string is given, it is converted to a `pywt.Wavelet`.
            Use `pywt.wavelist` to get a list of available wavelets.

            Possible wavelet families are:

            ``'haar'``: Haar

            ``'db'``: Daubechies

            ``'sym'``: Symlets

            ``'coif'``: Coiflets

            ``'bior'``: Biorthogonal

            ``'rbio'``: Reverse biorthogonal

            ``'dmey'``: Discrete FIR approximation of the Meyer wavelet

        variant : {'forward', 'inverse', 'adjoint'}
            Wavelet transform variant to be created.
        nlevels : positive int, optional
            Number of scaling levels to be used in the decomposition. The
            maximum number of levels can be calculated with
            `pywt.dwtn_max_level`.
            Default: Use maximum number of levels.
        pad_mode : string, optional
            Method to be used to extend the signal.

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not repeating the
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
            Only available with ``impl='pywt'``, See ``pywt.Modes.modes``.

            ``'reflect'``: Reflect at the boundary, without repeating the
            outmost values.

            ``'antisymmetric'``: Anti-symmetric variant of ``symmetric``.

            ``'antireflect'``: Anti-symmetric variant of ``reflect``.

            For reference, the following table compares the naming conventions
            for the modes in ODL vs. PyWavelets::

                ======================= ==================
                          ODL               PyWavelets
                ======================= ==================
                symmetric               symmetric
                reflect                 reflect
                order1                  smooth
                order0                  constant
                constant, pad_const=0   zero
                periodic                periodic
                pywt_per                periodization
                antisymmetric           antisymmetric
                antireflect             antireflect
                ======================= ==================

            See `signal extension modes`_ for an illustration of the modes
            (under the PyWavelets naming conventions).
        pad_const : float, optional
            Constant value to use if ``pad_mode == 'constant'``. Ignored
            otherwise. Constants other than 0 are not supported by the
            ``pywt`` back-end.
        impl : {'pywt'}, optional
            Back-end for the wavelet transform.
        axes : sequence of ints, optional
            Axes over which the DWT that created ``coeffs`` was performed.  The
            default value of ``None`` corresponds to all axes. When not all
            axes are included this is analagous to a batch transform in
            ``len(axes)`` dimensions looped over the non-transformed axes. In
            orther words, filtering and decimation does not occur along any
            axes not in ``axes``.

        References
        ----------
        .. _signal extension modes:
           https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
        """
        if not isinstance(space, DiscreteLp):
            raise TypeError('`space` {!r} is not a `DiscreteLp` instance.'
                            ''.format(space))

        self.__impl, impl_in = str(impl).lower(), impl
        if self.impl not in _SUPPORTED_WAVELET_IMPLS:
            raise ValueError("`impl` '{}' not supported".format(impl_in))

        if axes is None:
            axes = tuple(range(space.ndim))
        elif np.isscalar(axes):
            axes = (axes,)
        elif len(axes) > space.ndim:
                raise ValueError("Too many axes.")
        self.axes = tuple(axes)

        if nlevels is None:
            nlevels = pywt.dwtn_max_level(space.shape, wavelet, self.axes)
        self.__nlevels, nlevels_in = int(nlevels), nlevels
        if self.nlevels != nlevels_in:
            raise ValueError('`nlevels` must be integer, got {}'
                             ''.format(nlevels_in))

        self.__impl, impl_in = str(impl).lower(), impl
        if self.impl not in _SUPPORTED_WAVELET_IMPLS:
            raise ValueError("`impl` '{}' not supported".format(impl_in))

        self.__wavelet = getattr(wavelet, 'name', str(wavelet).lower())
        self.__pad_mode = str(pad_mode).lower()
        self.__pad_const = space.field.element(pad_const)

        if self.impl == 'pywt':
            self.pywt_pad_mode = pywt_pad_mode(pad_mode, pad_const)
            self.pywt_wavelet = pywt_wavelet(self.wavelet)
            # determine coefficient shapes (without running wavedecn)
            self._coeff_shapes = pywt.wavedecn_shapes(
                space.shape, wavelet, mode=self.pywt_pad_mode,
                level=self.nlevels, axes=self.axes)
            # precompute slices into the (raveled) coeffs
            self._coeff_slices = precompute_raveled_slices(self._coeff_shapes)
            coeff_size = pywt.wavedecn_size(self._coeff_shapes)
            coeff_space = space.tspace_type(coeff_size, dtype=space.dtype)
        else:
            raise RuntimeError("bad `impl` '{}'".format(self.impl))

        variant, variant_in = str(variant).lower(), variant
        if variant not in ('forward', 'inverse', 'adjoint'):
            raise ValueError("`variant` '{}' not understood"
                             "".format(variant_in))
        self.__variant = variant

        if variant == 'forward':
            super(WaveletTransformBase, self).__init__(
                domain=space, range=coeff_space, linear=True)
        else:
            super(WaveletTransformBase, self).__init__(
                domain=coeff_space, range=space, linear=True)

    @property
    def impl(self):
        """Implementation back-end of this wavelet transform."""
        return self.__impl

    @property
    def nlevels(self):
        """Number of scaling levels in this wavelet transform."""
        return self.__nlevels

    @property
    def wavelet(self):
        """Name of the wavelet used in this wavelet transform."""
        return self.__wavelet

    @property
    def pad_mode(self):
        """Padding mode used for extending input beyond its boundary."""
        return self.__pad_mode

    @property
    def pad_const(self):
        """Value for extension used in ``'constant'`` padding mode."""
        return self.__pad_const

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.pywt_wavelet.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.pywt_wavelet.biorthogonal

    def scales(self):
        """Get the scales of each coefficient.

        Returns
        -------
        scales : ``range`` element
            The scale of each coefficient, given by an integer. 0 for the
            lowest resolution and self.nlevels for the highest.
        """
        if self.impl == 'pywt':
            if self.__variant == 'forward':
                discr_space = self.domain
                wavelet_space = self.range
            else:
                discr_space = self.range
                wavelet_space = self.domain

            shapes = pywt.wavedecn_shapes(discr_space.shape, self.pywt_wavelet,
                                          mode=self.pywt_pad_mode,
                                          level=self.nlevels, axes=self.axes)
            coeff_list = [np.full(shapes[0], 0)]
            for i in range(1, 1 + len(shapes[1:])):
                coeff_list.append({k: np.full(shapes[i][k], i)
                                   for k in shapes[i].keys()})
            coeffs = pywt.ravel_coeffs(coeff_list, axes=self.axes)[0]
            return wavelet_space.element(coeffs)
        else:
            raise RuntimeError("bad `impl` '{}'".format(self.impl))


class WaveletTransform(WaveletTransformBase):

    """Discrete wavelet transform between discretized Lp spaces."""

    def __init__(self, domain, wavelet, nlevels=None, pad_mode='constant',
                 pad_const=0, impl='pywt', axes=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
        wavelet : string or `pywt.Wavelet`
            Specification of the wavelet to be used in the transform.
            If a string is given, it is converted to a `pywt.Wavelet`.
            Use `pywt.wavelist` to get a list of available wavelets.

            Possible wavelet families are:

            ``'haar'``: Haar

            ``'db'``: Daubechies

            ``'sym'``: Symlets

            ``'coif'``: Coiflets

            ``'bior'``: Biorthogonal

            ``'rbio'``: Reverse biorthogonal

            ``'dmey'``: Discrete FIR approximation of the Meyer wavelet

        nlevels : positive int, optional
            Number of scaling levels to be used in the decomposition. The
            maximum number of levels can be calculated with
            `pywt.dwtn_max_level`.
            Default: Use maximum number of levels.
        pad_mode : string, optional
            Method to be used to extend the signal.

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not repeating the
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
            Only available with ``impl='pywt'``, See ``pywt.Modes.modes``.

            ``'reflect'``: Reflect at the boundary, without repeating the
            outmost values.

            ``'antisymmetric'``: Anti-symmetric variant of ``symmetric``.

            ``'antireflect'``: Anti-symmetric variant of ``reflect``.

            For reference, the following table compares the naming conventions
            for the modes in ODL vs. PyWavelets::

                ======================= ==================
                          ODL               PyWavelets
                ======================= ==================
                symmetric               symmetric
                reflect                 reflect
                order1                  smooth
                order0                  constant
                constant, pad_const=0   zero
                periodic                periodic
                pywt_per                periodization
                antisymmetric           antisymmetric
                antireflect             antireflect
                ======================= ==================

            See `signal extension modes`_ for an illustration of the modes
            (under the PyWavelets naming conventions).
        pad_const : float, optional
            Constant value to use if ``pad_mode == 'constant'``. Ignored
            otherwise. Constants other than 0 are not supported by the
            ``pywt`` back-end.
        impl : {'pywt'}, optional
            Backend for the wavelet transform.
        axes : sequence of ints, optional
            Axes over which the DWT that created ``coeffs`` was performed.  The
            default value of ``None`` corresponds to all axes. When not all
            axes are included this is analagous to a batch transform in
            ``len(axes)`` dimensions looped over the non-transformed axes. In
            orther words, filtering and decimation does not occur along any
            axes not in ``axes``.

        Examples
        --------
        Compute a very simple wavelet transform in a discrete 2D space with
        4 sampling points per axis:

        >>> space = odl.uniform_discr([0, 0], [1, 1], (4, 4))
        >>> wavelet_trafo = odl.trafos.WaveletTransform(
        ...     domain=space, nlevels=1, wavelet='haar')
        >>> wavelet_trafo.is_biorthogonal
        True
        >>> data = [[1, 1, 1, 1],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 1, 1],
        ...         [1, 0, 1, 0]]
        >>> decomp = wavelet_trafo(data)
        >>> decomp.shape
        (16,)

        It is also possible to apply the transform only along a subset of the
        axes. Here, we apply a 1D wavelet transfrom along axis 0 for each
        index along axis 1:

        >>> wavelet_trafo = odl.trafos.WaveletTransform(
        ...     domain=space, nlevels=1, wavelet='haar', axes=(0,))
        >>> decomp = wavelet_trafo(data)
        >>> decomp.shape
        (16,)

        In general, the size of the coefficients may exceed the size of the
        input data when the wavelet is longer than the Haar wavelet. This
        due to extra coefficients that must be kept for perfect reconstruction.
        No extra boundary coefficients are needed when the edge mode is
        ``"pywt_periodic"`` and the size along each transformed axis is a
        multiple of ``2**nlevels``.

        >>> space = odl.uniform_discr([0, 0], [1, 1], (16, 16))
        >>> space.size
        256
        >>> wavelet_trafo = odl.trafos.WaveletTransform(
        ...     domain=space, nlevels=2, wavelet='db2',
        ...     pad_mode='pywt_periodic')
        >>> decomp = wavelet_trafo(np.ones(space.shape))
        >>> decomp.shape
        (256,)
        >>> wavelet_trafo = odl.trafos.WaveletTransform(
        ...     domain=space, nlevels=2, wavelet='db2', pad_mode='symmetric')
        >>> decomp = wavelet_trafo(np.ones(space.shape))
        >>> decomp.shape
        (387,)

        References
        ----------
        .. _signal extension modes:
           https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
        """
        super(WaveletTransform, self).__init__(
            space=domain, wavelet=wavelet, nlevels=nlevels, variant='forward',
            pad_mode=pad_mode, pad_const=pad_const, impl=impl, axes=axes)

    def _call(self, x):
        """Return wavelet transform of ``x``."""
        if self.impl == 'pywt':
            coeffs = pywt.wavedecn(
                x, wavelet=self.pywt_wavelet, level=self.nlevels,
                mode=self.pywt_pad_mode, axes=self.axes)
            return pywt.ravel_coeffs(coeffs, axes=self.axes)[0]
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
            if `is_orthogonal` is ``False``
        """
        if self.is_orthogonal:
            scale = 1 / self.domain.partition.cell_volume
            return scale * self.inverse
        else:
            # TODO: put adjoint here
            return super(WaveletTransform, self).adjoint

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
            range=self.domain, wavelet=self.pywt_wavelet, nlevels=self.nlevels,
            pad_mode=self.pad_mode, pad_const=self.pad_const, impl=self.impl,
            axes=self.axes)


class WaveletTransformInverse(WaveletTransformBase):

    """Discrete inverse wavelet trafo between discrete L2 spaces.

    See Also
    --------
    WaveletTransform
    """

    def __init__(self, range, wavelet, nlevels=None, pad_mode='constant',
                 pad_const=0, impl='pywt', axes=None):
        """Initialize a new instance.

         Parameters
        ----------
        range : `DiscreteLp`
            Domain of the forward wavelet transform (the "image domain"),
            which is the range of this inverse transform.
        wavelet : string or `pywt.Wavelet`
            Specification of the wavelet to be used in the transform.
            If a string is given, it is converted to a `pywt.Wavelet`.
            Use `pywt.wavelist` to get a list of available wavelets.

            Possible wavelet families are:

            ``'haar'``: Haar

            ``'db'``: Daubechies

            ``'sym'``: Symlets

            ``'coif'``: Coiflets

            ``'bior'``: Biorthogonal

            ``'rbio'``: Reverse biorthogonal

            ``'dmey'``: Discrete FIR approximation of the Meyer wavelet

        nlevels : positive int, optional
            Number of scaling levels to be used in the decomposition. The
            maximum number of levels can be calculated with
            `pywt.dwtn_max_level`.
            Default: Use maximum number of levels.
        pad_mode : string, optional
            Method to be used to extend the signal.

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not repeating the
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
            Only available with ``impl='pywt'``, See ``pywt.Modes.modes``.

            ``'reflect'``: Reflect at the boundary, without repeating the
            outmost values.

            ``'antisymmetric'``: Anti-symmetric variant of ``symmetric``.

            ``'antireflect'``: Anti-symmetric variant of ``reflect``.

            For reference, the following table compares the naming conventions
            for the modes in ODL vs. PyWavelets::

                ======================= ==================
                          ODL               PyWavelets
                ======================= ==================
                symmetric               symmetric
                reflect                 reflect
                order1                  smooth
                order0                  constant
                constant, pad_const=0   zero
                periodic                periodic
                pywt_per                periodization
                antisymmetric           antisymmetric
                antireflect             antireflect
                ======================= ==================

            See `signal extension modes`_ for an illustration of the modes
            (under the PyWavelets naming conventions).
        pad_const : float, optional
            Constant value to use if ``pad_mode == 'constant'``. Ignored
            otherwise. Constants other than 0 are not supported by the
            ``pywt`` back-end.
        impl : {'pywt'}, optional
            Back-end for the wavelet transform.
        axes : sequence of ints, optional
            Axes over which the DWT that created ``coeffs`` was performed.  The
            default value of ``None`` corresponds to all axes. When not all
            axes are included this is analagous to a batch transform in
            ``len(axes)`` dimensions looped over the non-transformed axes. In
            orther words, filtering and decimation does not occur along any
            axes not in ``axes``.

        Examples
        --------
        Check that the inverse is the actual inverse on a simple example on
        a discrete 2D space with 4 sampling points per axis:

        >>> space = odl.uniform_discr([0, 0], [1, 1], (4, 4))
        >>> wavelet_trafo = odl.trafos.WaveletTransform(
        ...     domain=space, nlevels=1, wavelet='haar')
        >>> orig_array = np.array([[1, 1, 1, 1],
        ...                        [0, 0, 0, 0],
        ...                        [0, 0, 1, 1],
        ...                        [1, 0, 1, 0]])
        >>> decomp = wavelet_trafo(orig_array)
        >>> recon = wavelet_trafo.inverse(decomp)
        >>> np.allclose(recon, orig_array)
        True

        References
        ----------
        .. _signal extension modes:
           https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
        """
        super(WaveletTransformInverse, self).__init__(
            space=range, wavelet=wavelet, variant='inverse', nlevels=nlevels,
            pad_mode=pad_mode, pad_const=pad_const, impl=impl, axes=axes)

    def _call(self, coeffs):
        """Return the inverse wavelet transform of ``coeffs``."""
        if self.impl == 'pywt':
            coeffs = pywt.unravel_coeffs(coeffs,
                                         coeff_slices=self._coeff_slices,
                                         coeff_shapes=self._coeff_shapes,
                                         output_format='wavedecn')
            recon = pywt.waverecn(
                coeffs, wavelet=self.pywt_wavelet, mode=self.pywt_pad_mode,
                axes=self.axes)
            recon_shape = self.range.shape
            if recon.shape != recon_shape:
                # If the original shape was odd along any transformed axes it
                # will have been rounded up to the next even size after the
                # reconstruction. The extra sample should be discarded.
                # The underlying reason is decimation by two in reconstruction
                # must keep ceil(N/2) samples in each band for perfect
                # reconstruction. Reconstruction then upsamples by two.
                # When N is odd, (2 * np.ceil(N/2)) != N.
                recon_slc = []
                for i, (n_recon, n_intended) in enumerate(zip(recon.shape,
                                                              recon_shape)):
                    if n_recon == n_intended + 1:
                        # Upsampling added one entry too much in this axis,
                        # drop last one
                        recon_slc.append(slice(-1))
                    elif n_recon == n_intended:
                        recon_slc.append(slice(None))
                    else:
                        raise ValueError(
                            'in axis {}: expected size {} or {} in '
                            '`recon_shape`, got {}'
                            ''.format(i, n_recon - 1, n_recon,
                                      n_intended))
                recon = recon[tuple(recon_slc)]
            return recon
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
            if `is_orthogonal` is ``False``

        See Also
        --------
        inverse
        """
        if self.is_orthogonal:
            scale = self.range.partition.cell_volume
            return scale * self.inverse
        else:
            # TODO: put adjoint here
            return super(WaveletTransformInverse, self).adjoint

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
            domain=self.range, wavelet=self.pywt_wavelet, nlevels=self.nlevels,
            pad_mode=self.pad_mode, pad_const=self.pad_const, impl=self.impl,
            axes=self.axes)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYWT_AVAILABLE)
