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
        self, space, samples, domain, range, skip_normalization=False):
        """Initialize a new instance.

        Parameters
        ----------
        space : DiscreteLp
            The uniform space in which the data lies
        samples : aray-like
            List of the fourier space positions where the coefficients are
            computed.
        domain : `TensorSpace`
            Domain of the non uniform FFT or its adjoint
        range : `TensorSpace`
            Range of the non uniform FFT or its adjoint
        skip_normalization : bool, optional
            Whether the samples normalization step should be skipped
        """
        super(NonUniformFourierTransformBase, self).__init__(
            domain=domain,
            range=range,
            linear=True,
        )
        self.space = space
        samples = np.asarray(samples, dtype=float)
        if samples.shape[1] != len(space.shape):
            raise ValueError(
                '`samples` dimensions incompatible with provided `shape`',
            )
        self.skip_normalization = skip_normalization
        self.samples = samples
        self.nfft = NFFT(N=space.shape, M=len(samples))
        self.adjoint_class = None
        self._has_run = False

    def _normalize(self):
        """Normalize samples in [-0.5; 0.5[.
        """
        if not self.skip_normalization:
            self.samples -= self.space.min_pt
            self.samples /= (self.space.max_pt - self.space.min_pt)
            self.samples -= 0.5
            self.samples[np.where(self.samples == 0.5)] = -0.5

class NonUniformFourierTransform(NonUniformFourierTransformBase):
    """Forward Non uniform Fast Fourier Transform.
    """
    def __init__(
        self, space, samples, skip_normalization=False):
        """Initialize a new instance.

        Parameters
        ----------
        space : DiscreteLp
            The uniform space in which the data lies
        samples : array-like
            List of the fourier space positions where the coefficients are
            computed.
        skip_normalization : bool, optional
            Whether the normalization step should be skipped
        """
        if not isinstance(space, DiscreteLp) or not space.is_uniform:
            raise ValueError("`space` should be a uniform `DiscreteLp`")
        super(NonUniformFourierTransform, self).__init__(
            space=space,
            samples=samples,
            domain=space,
            range=cn(len(samples)),
            skip_normalization=skip_normalization,
        )

    @property
    def adjoint(self):
        return NonUniformFourierTransformAdjoint(
            space=self.space,
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
        if not self._has_run:
            self._normalize()
            self.nfft.x = self.samples
            self.nfft.precompute()
            self._has_run = True
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
        self, space, samples, skip_normalization=False):
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
        """
        if not isinstance(space, DiscreteLp) or not space.is_uniform:
            raise ValueError("`space` should be a uniform `DiscreteLp`")
        super(NonUniformFourierTransformAdjoint, self).__init__(
            space=space,
            samples=samples,
            domain=cn(len(samples)),
            range=space,
            skip_normalization=skip_normalization,
        )

    @property
    def adjoint(self):
        return NonUniformFourierTransform(
            space=self.space,
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
        if not self._has_run:
            self._normalize()
            self.nfft.x = self.samples
            self.nfft.precompute()
            self._has_run = True
        self.nfft.f = np.asarray(x)
        out = self.nfft.adjoint()
        # The normalization is inspired from
        # https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/reconstruct/fourier.py#L123
        out /= np.sqrt(self.nfft.M)
        return out


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYNFFT_AVAILABLE)
