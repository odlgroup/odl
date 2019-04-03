# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Discretized non-uniform Fourier transform on L^p spaces."""

from __future__ import division
import numpy as np

from odl import DiscreteLp, cn
from odl.operator import Operator
from odl.trafos import PYNFFT_AVAILABLE
if PYNFFT_AVAILABLE:
    from pynfft.nfft import NFFT


class NonUniformFourierTransformBase(Operator):
    """Non uniform Fast Fourier Transform.
    """
    def __init__(
        self, shape, samples, domain, range, skip_normalization=False,
        max_frequencies=None):
        """Initialize a new instance.

        Parameters
        ----------
        shape : tuple
            The dimensions of the data whose non uniform FFT has to be
            computed
        samples : aray-like
            List of the fourier space positions where the coefficients are
            computed.
        domain : `DiscreteLp`
            Domain of the non uniform FFT or its adjoint
        range : `DiscreteLp`
            Range of the non uniform FFT or its adjoint
        skip_normalization : bool, optional
            Whether the samples normalization step should be skipped
        max_frequencies : None or int or float or `numpy.ndarray`, optional
            The max frequency for each dimension in the frequency space.
            If int or float, the max frequency will be the same for each
            dimension of the frequency space.
            If None, the max frequency for each dimension will be the max
            of the absolute value of the dimension for all the samples.
            Defaults to None.
        """
        super(NonUniformFourierTransformBase, self).__init__(
            domain=domain,
            range=range,
            linear=True,
        )
        self.shape = shape
        samples = np.asarray(samples, dtype=float)
        if samples.shape[1] != len(shape):
            raise ValueError(
                '`samples` dimensions incompatible with provided `shape`',
            )
        self.samples = self._normalize(
            samples,
            skip_normalization=skip_normalization,
            max_frequencies=max_frequencies,
        )
        self.nfft = NFFT(N=shape, M=len(samples))
        self.nfft.x = samples
        self.adjoint_class = None
        self._is_precomputed = False

    def _normalize(
        self, samples, skip_normalization=False, max_frequencies=None):
        """Normalize samples in [-0.5; 0.5[.

        Parameters
        ---------
        samples : `numpy.ndarray`
            The samples to be normalized
        skip_normalization : bool, optional
            Whether the normalization step should be skipped
        max_frequencies : None or int or float or `numpy.ndarray`, optional
            The max frequency for each dimension in the frequency space.
            If int or float, the max frequency will be the same for each
            dimension of the frequency space.
            If None, the max frequency for each dimension will be the max
            of the absolute value of the dimension for all the samples.
            Defaults to None.

        Returns
        -------
        samples : `numpy.ndarray`
            The normalized samples
        """
        if skip_normalization:
            return samples
        if max_frequencies is None:
            max_frequencies = np.max(np.abs(samples), axis=0)
        elif isinstance(max_frequencies, (int, float)):
            max_frequencies = max_frequencies * np.ones(samples.shape[1])
        samples /= max_frequencies
        samples -= 0.5
        samples[np.where(samples == 0.5)] = -0.5
        return samples


class NonUniformFourierTransform(NonUniformFourierTransformBase):
    """Forward Non uniform Fast Fourier Transform.
    """
    def __init__(
        self, space, samples, skip_normalization=False, max_frequencies=None):
        """Initialize a new instance.

        Parameters
        ----------
        shape : DiscreteLp
            The uniform space in which the data lies
        samples : array-like
            List of the fourier space positions where the coefficients are
            computed.
        skip_normalization : bool, optional
            Whether the normalization step should be skipped
        max_frequencies : None or int or float or `numpy.ndarray`, optional
            The max frequency for each dimension in the frequency space.
            If int or float, the max frequency will be the same for each
            dimension of the frequency space.
            If None, the max frequency for each dimension will be the max
            of the absolute value of the dimension for all the samples.
            Defaults to None.
        """
        if not isinstance(space, DiscreteLp) or not space.is_uniform:
            raise ValueError("`space` should be a uniform `DiscreteLp`")
        super(NonUniformFourierTransform, self).__init__(
            shape=space.shape,
            samples=samples,
            domain=space,
            range=cn(len(samples)),
            skip_normalization=skip_normalization,
            max_frequencies=max_frequencies,
        )
        self.adjoint_class = NonUniformFourierTransformAdjoint

    @property
    def adjoint(self):
        return NonUniformFourierTransformAdjoint(
            shape=self.shape,
            samples=self.samples,
            skip_normalization=True,
        )

    def _call(self, x):
        """Compute the direct non uniform FFT.

        Parameters
        ----------
        x : `numpy.ndarray`
            The data whose non uniform FFT you want to compute

        Returns
        -------
        out_normalized : `numpy.ndarray`
            Result of the transform
        """
        if not self._is_precomputed:
            self.nfft.precompute()
            self._is_precomputed = True
        self.nfft.f_hat = np.asarray(x)
        out = self.nfft.trafo()
        # The normalization is inspired from
        # https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/reconstruct/fourier.py#L123
        out /= np.sqrt(self.nfft.M)
        return out


class NonUniformFourierTransformAdjoint(NonUniformFourierTransformBase):
    """Adjoint of Non uniform Fast Fourier Transform.
    """
    def __init__(
        self, space, samples, skip_normalization=False, max_frequencies=None):
        """Initialize a new instance.

        Parameters
        ----------
        space : DiscreteLp
            The uniform space in which the data lies
        samples : aray-like
            List of the fourier space positions where the coefficients are
            computed.
        skip_normalization : bool, optional
            Whether the normalization step should be skipped
        max_frequencies : None or int or float or `numpy.ndarray`, optional
            The max frequency for each dimension in the frequency space.
            If int or float, the max frequency will be the same for each
            dimension of the frequency space.
            If None, the max frequency for each dimension will be the max
            of the absolute value of the dimension for all the samples.
            Defaults to None.
        """
        if not isinstance(space, DiscreteLp) or not space.is_uniform:
            raise ValueError("`space` should be a uniform `DiscreteLp`")
        super(NonUniformFourierTransformAdjoint, self).__init__(
            shape=space.shape,
            samples=samples,
            domain=cn(len(samples)),
            range=space,
            skip_normalization=skip_normalization,
            max_frequencies=max_frequencies,
        )
        self.adjoint_class = NonUniformFourierTransform

    @property
    def adjoint(self):
        return NonUniformFourierTransform(
            shape=self.shape,
            samples=self.samples,
            skip_normalization=True
        )

    def _call(self, x):
        """Compute the adjoint non uniform FFT.

        Parameters
        ----------
        x : `numpy.ndarray`
            The data whose non uniform FFT adjoint you want to compute

        Returns
        -------
        out_normalized : `numpy.ndarray`
            Result of the adjoint transform
        """
        if not self._is_precomputed:
            self.nfft.precompute()
            self._is_precomputed = True
        self.nfft.f = np.asarray(x)
        out = self.nfft.adjoint()
        # The normalization is inspired from
        # https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/reconstruct/fourier.py#L123
        out /= np.sqrt(self.nfft.M)
        return out


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYNFFT_AVAILABLE)
