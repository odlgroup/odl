# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""ODL integration with shearlab."""

import odl
import numpy as np
from threading import Lock

# Library for the shearlab library

import julia
import matplotlib.pyplot as plt
from numpy import ceil
from numpy.fft import fft2, ifft2, fftshift, ifftshift

__all__ = ('ShearlabOperator',)


class ShearlabOperator(odl.Operator):

    """Shearlet transform using Shearlab.jl as backend.
    This is the non-compact shearlet transform implemented using the Fourier
    transform.
    """

    def __init__(self, space, num_scales):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp`
            The space on which the shearlet transform should act. Must be
            two-dimensional.
        num_scales : nonnegative `int`
            The number of scales for the shearlet transform, higher numbers
            mean better edge resolution but more computational burden.

        Examples
        --------
        Create a 2d-shearlet transform:
        >>> space = odl.uniform_discr([-1, -1], [1, 1], [128, 128])
        >>> shearlet_transform = ShearlabOperator(space, num_scales=2)
        """
        self.shearlet_system = getshearletsystem2D(
            space.shape[0], space.shape[1], num_scales)
        range = space ** self.shearlet_system.nShearlets
        self.mutex = Lock()
        super(ShearlabOperator, self).__init__(space, range, True)

    def _call(self, x):
        """``self(x)``."""
        with self.mutex:
            result = sheardec2D(x, self.shearlet_system)
            return np.moveaxis(result, -1, 0)

    @property
    def adjoint(self):
        """The adjoint operator."""
        op = self

        class ShearlabOperatorAdjoint(odl.Operator):

            """Adjoint of the shearlet transform.

            See Also
            --------
            odl.contrib.shearlab.ShearlabOperator
            """

            def __init__(self):
                """Initialize a new instance."""
                self.mutex = op.mutex
                self.shearlet_system = op.shearlet_system
                super(ShearlabOperatorAdjoint, self).__init__(
                    op.range, op.domain, True)

            def _call(self, x):
                """``self(x)``."""
                with op.mutex:
                    x = np.moveaxis(x, 0, -1)
                    return sheardecadjoint2D(x, op.shearlet_system)

            @property
            def adjoint(self):
                """The adjoint operator."""
                return op

            @property
            def inverse(self):
                """The inverse operator."""
                op = self

                class ShearlabOperatorAdjointInverse(odl.Operator):

                    """
                    Adjoint of the inverse/Inverse of the adjoint
                    of shearlet transform.

                    See Also
                    --------
                    odl.contrib.shearlab.ShearlabOperator
                    """

                    def __init__(self):
                        """Initialize a new instance."""
                        self.mutex = op.mutex
                        self.shearlet_system = op.shearlet_system
                        super(ShearlabOperatorAdjointInverse, self).__init__(
                            op.range, op.domain, True)

                    def _call(self, x):
                        """``self(x)``."""
                        with op.mutex:
                            result = shearrecadjoint2D(x, op.shearlet_system)
                            return np.moveaxis(result, -1, 0)

                    @property
                    def adjoint(self):
                        """The adjoint operator."""
                        return op.adjoint.inverse

                    @property
                    def inverse(self):
                        """The inverse operator."""
                        return op

                return ShearlabOperatorAdjointInverse()

        return ShearlabOperatorAdjoint()

    @property
    def inverse(self):
        """The inverse operator."""
        op = self

        class ShearlabOperatorInverse(odl.Operator):

            """Inverse of the shearlet transform.

            See Also
            --------
            odl.contrib.shearlab.ShearlabOperator
            """

            def __init__(self):
                """Initialize a new instance."""
                self.mutex = op.mutex
                self.shearlet_system = op.shearlet_system
                super(ShearlabOperatorInverse, self).__init__(
                    op.range, op.domain, True)

            def _call(self, x):
                """``self(x)``."""
                with op.mutex:
                    x = np.moveaxis(x, 0, -1)
                    return shearrec2D(x, op.shearlet_system)

            @property
            def adjoint(self):
                """The inverse operator."""
                op = self

                class ShearlabOperatorInverseAdjoint(odl.Operator):

                    """
                    Adjoint of the inverse/Inverse of the adjoint
                    of shearlet transform.

                    See Also
                    --------
                    odl.contrib.shearlab.ShearlabOperator
                    """

                    def __init__(self):
                        """Initialize a new instance."""

                        self.mutex = op.mutex
                        self.shearlet_system = op.shearlet_system
                        super(ShearlabOperatorInverseAdjoint, self).__init__(
                            op.range, op.domain, True)

                    def _call(self, x):
                        """``self(x)``."""
                        with op.mutex:
                            result = shearrecadjoint2D(x, op.shearlet_system)
                            return np.moveaxis(result, -1, 0)

                    @property
                    def adjoint(self):
                        """The adjoint operator."""
                        return op

                    @property
                    def inverse(self):
                        """The inverse operator."""
                        return op.inverse.adjoint

                return ShearlabOperatorInverseAdjoint()

            @property
            def inverse(self):
                """The inverse operator."""
                return op

        return ShearlabOperatorInverse()


# Python library for shearlab.jl

def load_julia_with_Shearlab():
    """Function to load Shearlab."""
    # Importing base
    j = julia.Julia()
    j.eval('using Shearlab')
    j.eval('using PyPlot')
    j.eval('using Images')
    return j


j = load_julia_with_Shearlab()


def load_image(name, n, m=None, gpu=None, square=None):
    """Function to load images with certain size."""
    if m is None:
        m = n
    if gpu is None:
        gpu = 0
    if square is None:
        square = 0
    command = ('Shearlab.load_image("{}", {}, {}, {}, {})'.format(name,
               n, m, gpu, square))
    return j.eval(command)


def imageplot(f, str=None, sbpt=None):
    """Plot an image generated by the library."""
    # Function to plot images
    if str is None:
        str = ''
    if sbpt is None:
        sbpt = []
    if sbpt != []:
        plt.subplot(sbpt[0], sbpt[1], sbpt[2])
    imgplot = plt.imshow(f, interpolation='nearest')
    imgplot.set_cmap('gray')
    plt.axis('off')
    if str != '':
        plt.title(str)


class Shearletsystem2D:
    """Class of shearlet system in 2D."""
    def __init__(self, shearlets, size, shearLevels, full, nShearlets,
                 shearletIdxs, dualFrameWeights, RMS, isComplex):
        self.shearlets = shearlets
        self.size = size
        self.shearLevels = shearLevels
        self.full = full
        self.nShearlets = nShearlets
        self.shearletIdxs = shearletIdxs
        self.dualFrameWeights = dualFrameWeights
        self.RMS = RMS
        self.isComplex = isComplex


def getshearletsystem2D(rows, cols, nScales, shearLevels=None,
                        full=None,
                        directionalFilter=None,
                        quadratureMirrorFilter=None):
    """Function to generate de 2D system."""
    if shearLevels is None:
        shearLevels = [float(ceil(i / 2)) for i in range(1, nScales + 1)]
    if full is None:
        full = 0
    if directionalFilter is None:
        directionalFilter = 'Shearlab.filt_gen("directional_shearlet")'
    if quadratureMirrorFilter is None:
        quadratureMirrorFilter = 'Shearlab.filt_gen("scaling_shearlet")'
    j.eval('rows=' + str(rows))
    j.eval('cols=' + str(cols))
    j.eval('nScales=' + str(nScales))
    j.eval('shearLevels=' + str(shearLevels))
    j.eval('full=' + str(full))
    j.eval('directionalFilter=' + directionalFilter)
    j.eval('quadratureMirrorFilter=' + quadratureMirrorFilter)
    j.eval('shearletsystem=Shearlab.getshearletsystem2D(rows, '
           'cols, nScales, shearLevels, full, directionalFilter, '
           'quadratureMirrorFilter) ')
    shearlets = j.eval('shearletsystem.shearlets')
    size = j.eval('shearletsystem.size')
    shearLevels = j.eval('shearletsystem.shearLevels')
    full = j.eval('shearletsystem.full')
    nShearlets = j.eval('shearletsystem.nShearlets')
    shearletIdxs = j.eval('shearletsystem.shearletIdxs')
    dualFrameWeights = j.eval('shearletsystem.dualFrameWeights')
    RMS = j.eval('shearletsystem.RMS')
    isComplex = j.eval('shearletsystem.isComplex')
    j.eval('shearletsystem = 0')
    return Shearletsystem2D(shearlets, size, shearLevels, full, nShearlets,
                            shearletIdxs, dualFrameWeights, RMS, isComplex)


def sheardec2D(X, shearletsystem):
    """Shearlet Decomposition function."""
    coeffs = np.zeros(shearletsystem.shearlets.shape, dtype=complex)
    Xfreq = fftshift(fft2(ifftshift(X)))
    for i in range(shearletsystem.nShearlets):
        coeffs[:, :, i] = fftshift(ifft2(ifftshift(Xfreq * np.conj(
                                   shearletsystem.shearlets[:, :, i]))))
    return coeffs.real


def shearrec2D(coeffs, shearletsystem):
    """Shearlet Recovery function."""
    X = np.zeros(coeffs.shape[:2], dtype=complex)
    for i in range(shearletsystem.nShearlets):
        X = X + fftshift(fft2(
            ifftshift(coeffs[:, :, i]))) * shearletsystem.shearlets[:, :, i]
    return (fftshift(ifft2(ifftshift((
            X / shearletsystem.dualFrameWeights))))).real


def sheardecadjoint2D(coeffs, shearletsystem):
    """Shearlet Decomposition adjoint function."""
    X = np.zeros(coeffs.shape[:2], dtype=complex)
    for i in range(shearletsystem.nShearlets):
        X = X + fftshift(fft2(
            ifftshift(coeffs[:, :, i]))) * np.conj(
            shearletsystem.shearlets[:, :, i])
    return (fftshift(ifft2(ifftshift(
            X / shearletsystem.dualFrameWeights)))).real


def shearrecadjoint2D(X, shearletsystem):
    """Shearlet Recovery adjoint function."""
    coeffs = np.zeros(shearletsystem.shearlets.shape, dtype=complex)
    Xfreq = fftshift(fft2(ifftshift(X)))
    for i in range(shearletsystem.nShearlets):
        coeffs[:, :, i] = fftshift(ifft2(ifftshift(
            Xfreq * shearletsystem.shearlets[:, :, i])))
    return coeffs.real


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
