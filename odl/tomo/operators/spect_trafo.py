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
"""Single-photon emission computerized tomography (SPECT) projectors """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

try:
    from NiftyPy import NiftyRec
    NIFTYREC_AVAILABLE = True
except ImportError:
    NIFTYREC_AVAILABLE = False

import numpy as np

from odl.operator import Operator
from odl import uniform_discr_frompartition
from odl.discr.lp_discr import DiscreteLp
from odl.tomo.geometry.spect import ParallelHoleCollimatorGeometry


__all__ = ('AttenuatedRayTransform', 'AttenuatedRayBackprojection',
           'NIFTYREC_AVAILABLE')
_SUPPORTED_IMPL = ('niftyrec_cpu', 'niftyrec_gpu')


class AttenuatedRayTransform(Operator):

    """The discrete attenuated Ray transform between L^p spaces.

    The attenuated Ray transform is a variant of the classical
    Ray transform  in which functions are integrated over lines
    with respect to an exponential weight given an attenuation map.
    """

    def __init__(self, dom, geometry, attenuation=None,
                 impl='niftyrec', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `DiscreteLp`
            Domain of the forward projector, 3 dimensional.
        geometry : `ParallelHoleCollimatorGeometry`
            The geometry of the transform.
        attenuation : domain `element-like`, optional
            Linear attenuation map for SPECT.
        impl : {'niftyrec_cpu', 'niftyrec_gpu'}, optional
            Implementation back-end for the transform. Supported back-ends:

            'niftyrec_cpu' : `NiftyRec`_ tomography toolbox using the
            `NiftyPy`_ Python wrapper and CPU

            'niftyrec_gpu' : `NiftyRec`_ tomography toolbox using the
            `NiftyPy`_ Python wrapper and GPU
        psf : `array-like` of small size, optional
            Point Spread Functions to be passed to NiftyRec
            for the correction of the collimator-detector response
            `shape` of `psf` typically is smaller than `dom.shape`

        References
        ----------
        .._NiftyRec: http://niftyrec.scienceontheweb.net/wordpress/
        .._NiftyPy: https://github.com/spedemon/niftypy
        """
        if not isinstance(dom, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp`'
                            ' instance'.format(dom))

        if not isinstance(geometry, ParallelHoleCollimatorGeometry):
            raise TypeError('geometry {!r} is not a'
                            '`ParallelHoleCollimatorGeometry` instance'
                            ''.format(geometry))

        ran = uniform_discr_frompartition(geometry.partition)
        super().__init__(dom, ran, linear=True)

        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_IMPL:
            raise ValueError('implementation {!r} not supported'
                             ''.format(impl_in))

        if impl.startswith('niftyrec'):
            if not NIFTYREC_AVAILABLE:
                raise ValueError('NiftyRec back-end not available')
            if impl == 'niftyrec_gpu' and dom.shape != (128, 128, 128):
                raise NotImplementedError("'niftyrec_gpu' back-end only "
                                          "supports volume size "
                                          "(128, 128, 128), got {}"
                                          "".format(dom.shape))
            self._use_gpu = (impl == 'niftyrec_gpu')
        else:
            raise RuntimeError('unknown implementation')

        self._geometry = geometry

        psf = kwargs.pop('psf', None)
        if psf is not None:
            self._psf = np.asarray(psf)
        else:
            self._psf = psf
        if attenuation is not None:
            self._attenuation = self.domain.element(attenuation)
        else:
            self._attenuation = attenuation
        self._impl = impl

    def _call(self, x):
        """Compute the discrete SPECT back-projection.

        Parameters
        ----------
        x : `DiscreteLpVector`
           The volume that is forward projected. The volume is an element
           in the domain of the operator.

        Returns
        -------
        out : ``range`` element
        """
        if self._impl.startswith('niftyrec'):
            return self._call_niftyrec(x.asarray())
        else:
            raise RuntimeError('unknown implementation')

    def _call_niftyrec(self, x):
        """Return ``self(x)`` for niftyrec back-end.

        Parameters
        ----------
        x :  `array-like`
           The volume that is forward projected. The volume is an element
           in the domain of the operator.

        Returns
        -------
        out : ``range`` element-like
        """
        if self._attenuation is None:
            attenuation = None
        else:
            attenuation = np.asfortranarray(self._attenuation, dtype='float32')
            attenuation *= self.domain.cell_sides[0]

        if self._psf is None:
            psf = None
        else:
            psf = np.asfortranarray(self._psf, dtype='float32')

        x = np.asfortranarray(x, dtype='float32')
        x *= self.domain.cell_sides[0]
        cameras = np.float32(self._geometry.angles[:, None])
        projection = NiftyRec.SPECT_project_parallelholes(
            x, cameras, attenuation, psf, use_gpu=self._use_gpu)
        projection = np.transpose(projection, (2, 0, 1))
        return projection

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return AttenuatedRayBackprojection(ran=self.domain,
                                           geometry=self._geometry,
                                           attenuation=self._attenuation,
                                           psf=self._psf,
                                           impl=self._impl)


class AttenuatedRayBackprojection(Operator):

    """The adjoint of the discrete attenuated Ray transform."""

    def __init__(self, ran, geometry, attenuation=None,
                 impl='niftyrec_cpu', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `DiscreteLp`
            Domain of the forward projector, 3 dimensional.
        geometry : `ParallelHoleCollimatorGeometry`
            The geometry of the transform.
        attenuation : domain `element-like`, optional
            Linear attenuation map for SPECT.
        impl : {'niftyrec_cpu', 'niftyrec_gpu'}, optional
            Implementation back-end for the transform. Supported back-ends:

            'niftyrec_cpu' : `NiftyRec`_ tomography toolbox using the
            `NiftyPy`_ Python wrapper and CPU

            'niftyrec_gpu' : `NiftyRec`_ tomography toolbox using the
            `NiftyPy`_ Python wrapper and GPU
        psf : `array` of small size, optional
            Point Spread Functions to be passed to NiftyRec
            for the correction of the collimator-detector response
            `shape` of `psf` typically is smaller than `dom.shape`

        References
        ----------
        .._NiftyRec: http://niftyrec.scienceontheweb.net/wordpress/

        .._NiftyPy: https://github.com/spedemon/niftypy
        """
        if not isinstance(ran, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp`'
                            ' instance'.format(ran))

        if not isinstance(geometry, ParallelHoleCollimatorGeometry):
            raise TypeError('geometry {!r} is not a'
                            '`ParallelHoleCollimatorGeometry` instance'
                            ''.format(geometry))

        dom = uniform_discr_frompartition(geometry.partition)
        super().__init__(dom, ran, linear=True)

        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_IMPL:
            raise ValueError('implementation {!r} not supported'
                             ''.format(impl_in))

        if impl.startswith('niftyrec'):
            if not NIFTYREC_AVAILABLE:
                raise ValueError('NiftyRec back-end not available')
            if impl == 'niftyrec_gpu' and ran.shape != (128, 128, 128):
                raise NotImplementedError('`niftyrec_gpu` back-end only '
                                          'supports volume size '
                                          '(128, 128, 128), got {}'
                                          ''.format(ran.shape))
            self._use_gpu = (impl == 'niftyrec_gpu')
        else:
            raise RuntimeError('unknown implementation')

        self._geometry = geometry

        psf = kwargs.pop('psf', None)
        if psf is not None:
            self._psf = np.asarray(psf)
        else:
            self._psf = psf
        if attenuation is not None:
            self._attenuation = self.range.element(attenuation)
        else:
            self._attenuation = attenuation
        self._impl = impl

    def _call(self, x):
        """Compute the discrete SPECT back-projection.

        Parameters
        ----------
        x : `DiscreteLpVector`
           The volume that is back-projected. The volume is an element
           in the range of the operator.

        Returns
        -------
        out : ``range`` element
        """
        if self._impl.startswith('niftyrec'):
            return self._call_niftyrec(x.asarray())
        else:
            raise RuntimeError('unknown implementation')

    def _call_niftyrec(self, x):
        """Return ``self(x)`` for niftyrec back-end.

        Parameters
        ----------
        x :  `array-like`
           The volume that is back-projected. The volume is an element
           in the range of the operator.

        Returns
        -------
        out : ``range`` element-like
        """
        if self._attenuation is None:
            attenuation = None
        else:
            attenuation = np.asfortranarray(self._attenuation, dtype='float32')
            attenuation *= self.domain.cell_sides[0]

        if self._psf is None:
            psf = None
        else:
            psf = np.asfortranarray(self._psf, dtype='float32')

        x = np.transpose(x, (1, 2, 0))
        x = np.asfortranarray(x, dtype='float32')
        cameras = np.float32(self._geometry.angles[:, None])
        activity = NiftyRec.SPECT_backproject_parallelholes(
            x, cameras, attenuation, psf, use_gpu=self._use_gpu)
        scale = self._geometry.det_partition.cell_volume
        scale *= self._geometry.motion_partition.cell_volume
        scale /= self.range.partition.cell_volume
        scale *= self.range.cell_sides[0]
        activity *= scale

        return activity

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return AttenuatedRayTransform(dom=self.range,
                                      geometry=self._geometry,
                                      attenuation=self._attenuation,
                                      psf=self._psf, impl=self._imlp)
