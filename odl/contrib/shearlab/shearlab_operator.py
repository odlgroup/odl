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
# import shearlab
from threading import Lock

# Library for the shearlab library

import julia
import matplotlib.pyplot as plt
import pylab
from math import ceil
from numpy.fft import fft2, ifft2, fftshift, ifftshift


__all__ = ('ShearlabOperator',)

class ShearlabOperator(odl.Operator):

    """Shearlet transform using Shearlab.jl as backend.
    This is the non-compact shearlet transform implemented using the fourier
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
        """Return ``self(x)``."""
        with self.mutex:
            result = sheardec2D(x, self.shearlet_system)
            return np.moveaxis(result, -1, 0)

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return ShearlabOperatorAdjoint(self)

    @property
    def inverse(self):
        """Return the inverse operator."""
        return ShearlabOperatorInverse(self)


class ShearlabOperatorAdjoint(odl.Operator):

    """Adjoint of the shearlet transform.
    Should not be used independently.
    See Also
    --------
    odl.contrib.shearlab.ShearlabOperator
    """

    def __init__(self, op):
        """Initialize a new instance.
        Parameters
        ----------
        op : `ShearlabOperator`
            The operator which this should be the adjoint of.
        """
        self.op = op
        super(ShearlabOperatorAdjoint, self).__init__(
            op.range, op.domain, True)

    def _call(self, x):
        """Return ``self(x)``."""
        with self.op.mutex:
            x = np.moveaxis(x, 0, -1)
            return sheardecadjoint2D(x, self.op.shearlet_system)

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return self.op

    @property
    def inverse(self):
        """Return the inverse operator."""
        return ShearlabOperatorAdjointInverse(self.op)

class ShearlabOperatorInverse(odl.Operator):

    """Inverse of the shearlet transform.
    Should not be used independently.
    See Also
    --------
    odl.contrib.shearlab.ShearlabOperator
    """

    def __init__(self, op):
        """Initialize a new instance.
        Parameters
        ----------
        op : `ShearlabOperator`
            The operator which this should be the inverse of.
        """
        self.op = op
        super(ShearlabOperatorInverse, self).__init__(
            op.range, op.domain, True)

    def _call(self, x):
        """Return ``self(x)``."""
        with self.op.mutex:
            x = np.moveaxis(x, 0, -1)
            return shearrec2D(x, self.op.shearlet_system)

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return ShearlabOperatorAdjointInverse(self.op)

    @property
    def inverse(self):
        """Return the inverse operator."""
        return self.op

class ShearlabOperatorAdjointInverse(odl.Operator):

    """Adjoint of the inverse/Inverse of the adjoint of shearlet transform.
    Should not be used independently.
    See Also
    --------
    odl.contrib.shearlab.ShearlabOperator
    """

    def __init__(self, op):
        """Initialize a new instance.
        Parameters
        ----------
        op : `ShearlabOperator`
            The operator which this should be the inverse of the adjoint of.
        """
        self.op = op
        super(ShearlabOperatorAdjointInverse, self).__init__(
            op.domain, op.range, True)

    def _call(self, x):
        """Return ``self(x)``."""
        with self.op.mutex:
            result = shearrecadjoint2D(x, self.op.shearlet_system)
            return np.moveaxis(result, -1, 0)

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return self.op.inverse

    @property
    def inverse(self):
        """Return the inverse operator."""
        return self.op.adjoint

## Python library for shearlab.jl

# Function to load Shearlab
def load_Shearlab():
    # Importing base
    j = julia.Julia()
    j.eval('using Shearlab')
    j.eval('using PyPlot')
    j.eval('using Images')
    return j
        
# Load Shearlab
j = load_Shearlab()

# Function to load images with certain size
def load_image(name, n, m=None, gpu=0, square=0):
    if m is None:
    	m = n
    command = 'Shearlab.load_image('+'"'+name+'"'+','+str(n)+','
    command = command + str(m)+','+str(gpu)+','+str(square)+')'
    return j.eval(command)

# Function to plot images
def imageplot(f, str='', sbpt=[]):
    """
        Use nearest neighbor interpolation for the display.
    """
    if sbpt != []:
        plt.subplot(sbpt[0], sbpt[1], sbpt[2])
    imgplot = plt.imshow(f, interpolation='nearest')
    imgplot.set_cmap('gray')
    pylab.axis('off')
    if str != '':
        plt.title(str)

# Class of shearlet system in 2D
class Shearletsystem2D:
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


# Function to generate de 2D system
def getshearletsystem2D(rows,cols,nScales,
                                shearLevels= None,
                                full= 0,
                                directionalFilter = 'Shearlab.filt_gen("directional_shearlet")',
                                quadratureMirrorFilter= 'Shearlab.filt_gen("scaling_shearlet")'):
    if shearLevels is None:
    	shearLevels = [float(ceil(i/2)) for i in range(1,nScales+1)]
    j.eval('rows='+str(rows))
    j.eval('cols='+str(cols))
    j.eval('nScales='+str(nScales))
    j.eval('shearLevels='+str(shearLevels))
    j.eval('full='+str(full))
    j.eval('directionalFilter='+directionalFilter)
    j.eval('quadratureMirrorFilter='+quadratureMirrorFilter)
    j.eval('shearletSystem = Shearlab.getshearletsystem2D(rows,cols,nScales,shearLevels,full,directionalFilter,quadratureMirrorFilter);')
    shearlets = j.eval('shearletSystem.shearlets')
    size = j.eval('shearletSystem.size')
    shearLevels = j.eval('shearletSystem.shearLevels')
    full = j.eval('shearletSystem.full')
    nShearlets = j.eval('shearletSystem.nShearlets')
    shearletIdxs = j.eval('shearletSystem.shearletIdxs')
    dualFrameWeights = j.eval('shearletSystem.dualFrameWeights')
    RMS = j.eval('shearletSystem.RMS')
    isComplex = j.eval('shearletSystem.isComplex');
    return  Shearletsystem2D(shearlets, size, shearLevels, full, nShearlets,
                                  shearletIdxs, dualFrameWeights, RMS, isComplex)

# Shearlet Decomposition function
def sheardec2D(X,shearletSystem):
    coeffs = np.zeros(shearletSystem.shearlets.shape,dtype=np.complex_)
    Xfreq = fftshift(fft2(ifftshift(X)))
    for i in range(shearletSystem.nShearlets):
        coeffs[:,:,i] = fftshift(ifft2(ifftshift(Xfreq*np.conj(shearletSystem.shearlets[:,:,i]))))
    return coeffs.real

# Shearlet Recovery function
def shearrec2D(coeffs,shearletSystem):
    X = np.zeros((coeffs.shape[0],coeffs.shape[1]),dtype=np.complex_)
    for i in range(shearletSystem.nShearlets):
        X = X+fftshift(fft2(ifftshift(coeffs[:,:,i])))*shearletSystem.shearlets[:,:,i]
    return (fftshift(ifft2(ifftshift((1/shearletSystem.dualFrameWeights)*X)))).real

# Shearlet Decomposition adjoint function
def sheardecadjoint2D(coeffs, shearletSystem):
    X = np.zeros((coeffs.shape[0], coeffs.shape[1]), dtype=complex)
    for i in range(shearletSystem.nShearlets):
        X = X+fftshift(fft2(ifftshift(coeffs[:,:,i])))*np.conj(shearletSystem.shearlets[:,:,i])
    return fftshift(ifft2(ifftshift(X))).real

# Shearlet Recovery adjoint function
def shearrecadjoint2D(X, shearletSystem):
    coeffs = np.zeros(shearletSystem.shearlets.shape,dtype=np.complex_)
    Xfreq = fftshift(fft2(ifftshift(X)))
    for i in range(shearletSystem.nShearlets):
        coeffs[:,:,i] = fftshift(ifft2(ifftshift(Xfreq*shearletSystem.shearlets[:,:,i])))
    return coeffs.real

    
if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
