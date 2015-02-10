# -*- coding: utf-8 -*-
"""
ctf.py -- contrast transfer function in electron tomography

Copyright 2014 Holger Kohr

This file is part of tomok.

tomok is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tomok is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with tomok.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object

import numpy as np


class ContrTransFunc(object):
    """Callable Contrast Transfer Function class.
    TODO: finish this properly."""

    def __init__(self, emcfg):

        self.osc_polycoeff = emcfg.osc_polycoeff
        self.env_polycoeff = emcfg.env_polycoeff
        self.cutoff2 = (emcfg.wavenum * emcfg.aperture / (emcfg.focal_len *
                                                          emcfg.magnif))**2

    def __call__(self, freq2, envelope=True):

        ctfval = np.exp(np.polyval(1j * self.osc_polycoeff, freq2))
        if envelope:
            ctfval *= np.exp(-np.polyval(self.env_polycoeff, freq2))

        return np.where(freq2 < self.cutoff2, ctfval, 0.0)

    # TODO: display method


class ContrTransFuncACR(object):
    """Callable class for the constant acr CTF.
    TODO: finish this."""

    def __init__(self, emcfg, acr=0.1):

        ocoeff = emcfg.osc_polycoeff
        ocoeff[3] = np.arctan(acr)
        self.osc_polycoeff = ocoeff
        self.env_polycoeff = emcfg.env_polycoeff
        self.cutoff2 = (emcfg.wavenum * emcfg.aperture / (emcfg.focal_len *
                                                          emcfg.magnif))**2

    def __call__(self, freq2, envelope=True):

        ctfval = np.sin(np.polyval(self.osc_polycoeff, freq2))
        if envelope:
            ctfval *= np.exp(-np.polyval(self.env_polycoeff, freq2))

        return np.where(freq2 < self.cutoff2, ctfval, 0.0)

    def zeros(self, num=0, maxfreq2=None):
        """The zeros as an array.
        TODO: finish"""

        # The sine zeros are those of the polynomials a*x^2 + b*x + c_i,
        # where a and b are the quadratic / linear coefficients of
        # the sine argument and c_i = constant coeff. - (i+1)*pi

        zeros = []
        p_a = self.osc_polycoeff[1]
        p_b = self.osc_polycoeff[2]

        maxzeros = 1000
        nmax = num if num else maxzeros

        for i in range(nmax):
            p_c = self.osc_polycoeff[3] - (i + 1) * np.pi
            zero = np.sqrt(p_b**2 - 4. * p_a * p_c) / (2 * p_a)
            if maxfreq2 is not None and zero > maxfreq2:
                break
            zeros.append(zero)

        return np.asarray(zeros)

    # TODO: display method
