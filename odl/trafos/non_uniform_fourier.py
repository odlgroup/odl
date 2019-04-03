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

from odl.discr import discr_sequence_space
from odl.operator import Operator
from odl.trafos import PYNFFT_AVAILABLE
if PYNFFT_AVAILABLE:
    from pynfft.nfft import NFFT

class NonUniformFourierTransformBase(Operator):
    """Non uniform Fast Fourier Transform.

    The normalization is inspired from pysap-mri, mainly this class:
    https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/reconstruct/fourier.py#L123
    """
    def __init__(self, shape, samples, domain, range):
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
        self.samples = samples
        self.nfft = NFFT(N=shape, M=len(samples))
        self.nfft.x = samples
        self.adjoint_class = None
        self._is_precomputed = False


class NonUniformFourierTransform(NonUniformFourierTransformBase):
    """Forward Non uniform Fast Fourier Transform.
    """
    def __init__(self, shape, samples):
        """Initialize a new instance.

        Parameters
        ----------
        shape : tuple
            The dimensions of the data whose non uniform FFT has to be
            computed
        samples : array-like
            List of the fourier space positions where the coefficients are
            computed.
        """
        super(NonUniformFourierTransform, self).__init__(
            shape=shape,
            samples=samples,
            domain=discr_sequence_space(shape, dtype=np.complex128),
            range=discr_sequence_space(
                [len(samples)],
                dtype=np.complex128,
            ),
        )
        self.adjoint_class = NonUniformFourierTransformAdjoint

    @property
    def adjoint(self):
        return NonUniformFourierTransformAdjoint(
            shape=self.shape,
            samples=self.samples,
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
        # The normalization is inspired from https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/reconstruct/fourier.py#L123
        out /= np.sqrt(self.nfft.M)
        return out


class NonUniformFourierTransformAdjoint(NonUniformFourierTransformBase):
    """Adjoint of Non uniform Fast Fourier Transform.
    """
    def __init__(self, shape, samples):
        """Initialize a new instance.

        Parameters
        ----------
        shape : tuple
            The dimensions of the data whose non uniform FFT adjoint has to be
            computed
        samples : aray-like
            List of the fourier space positions where the coefficients are
            computed.
        """
        super(NonUniformFourierTransformAdjoint, self).__init__(
            shape=shape,
            samples=samples,
            domain=discr_sequence_space(
                [len(samples)],
                dtype=np.complex128,
            ),
            range=discr_sequence_space(shape, dtype=np.complex128),
        )
        self.adjoint_class = NonUniformFourierTransform

    @property
    def adjoint(self):
        return NonUniformFourierTransform(
            shape=self.shape,
            samples=self.samples,
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
        # The normalization is inspired from https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/reconstruct/fourier.py#L123
        out /= np.sqrt(self.nfft.M)
        return out

if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYNFFT_AVAILABLE)
