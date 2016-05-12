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

# External
try:
    from NiftyPy.NiftyRec import (SPECT_project_parallelholes,
                                  SPECT_backproject_parallelholes)
    NIFTYREC_AVAILABLE = True
except ImportError:
    NIFTYREC_AVAILABLE = False

import numpy as np

# Internal
import odl
from odl.operator.operator import Operator


_SUPPORTED_IMPL = ('niftyrec')
__all__ = ('SpectProject', 'SpectBackproject')


class SpectProject(Operator):
    """Spect projector using backend NiftyRec"""
    def __init__(self, dom, geometry, attenuation, psf, use_gpu=False,
                 impl='niftyrec', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `DiscreteLp`
            Discretized space, the domain of the forward projector
        geometry : `Geometry`
            The geometry of the transform. An ODL geometry instance
            that contains information about the operator domain
        attenuation : `DiscreteLpVector`
            Linear attenuation map for SPECT.
        psf : `DiscreteLpVector`
            Point Spread Functions for the correction of the
            collimator-detector response
        use_gpu : `bool`, optional
            'False' if not using GPU
            `True` if using GPU provided that `impl` supports it
        impl='niftyrec', optional
            Implementation back-end for the transform. Supported back-ends:
            NiftyRec Tomography Toolbox through NiftyPy a Python wrapper
            for NiftyRec
        """
        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_IMPL:
            raise ValueError('implementation {!r} not supported.'
                             ''.format(impl_in))

        self.geometry = geometry
        det_nx = geometry.det_grid.shape[0]
        det_ny = geometry.det_grid.shape[1]
        startAngle = float(geometry.motion_grid.points()[0])
        stopAngle = float(geometry.motion_grid.points()[-1]) * (180 / np.pi)
        N_proj = len(geometry.motion_grid.points())

        self._cameras = np.float32(np.linspace(startAngle, stopAngle, N_proj,
                                   dtype=np.float32).reshape((N_proj, 1)))
        self._use_gpu = use_gpu
        self._attenuation = attenuation
        self._psf = psf

        lowleft_x = geometry.detector.params.corners()[0][0]
        lowleft_y = geometry.detector.params.corners()[0][1]
        upright_x = geometry.detector.params.corners()[-1][0]
        upright_y = geometry.detector.params.corners()[-1][1]
        ran = odl.uniform_discr([lowleft_x, lowleft_y, startAngle],
                                [upright_x, upright_y, stopAngle],
                                [det_nx, det_ny, N_proj])

        super().__init__(dom, ran, linear=True)

    def _call(self, x):
        """Compute the discrete SPECT ptojection

        Parameters
        ----------
        x : `DiscreteLpVector`
           Element in the domain of the operator to be forward projected

        Returns
        -------
        out : `DiscreteLpVector`
            Returns an element in the projection space
        """
        if self._attenuation is None:
            attenuation = None
        else:
            attenuation = np.float32(
                np.asfortranarray(self._attenuation.asarray()))

        if self._psf is None:
            psf = None
        else:
            psf = np.float32(self._psf.asarray())

        x = np.float32(np.asfortranarray(x.asarray()))
        projection = SPECT_project_parallelholes(x, self._cameras,
                                                 attenuation, psf,
                                                 use_gpu=self._use_gpu)
        return self.range.element(projection)

    @property
    def adjoint(self):
        """The back-projector associated with this array."""
        return SpectBackproject(ran=self.domain,
                                geometry=self.geometry,
                                attenuation=self._attenuation, psf=self._psf,
                                use_gpu=self._use_gpu)


class SpectBackproject(Operator):

    def __init__(self, ran, geometry, attenuation, psf, use_gpu=False,
                 impl='niftyrec', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        ran : `DiscreteLp`
            Reconstruction space, the range of the back-projector
        geometry : `Geometry`
            The geometry of the transform. An ODL geometry instance
            that contains information about the operator domain
        attenuation : element in :class:`~odl.discr.lp_discr.DiscreteLp`
            Linear attenuation map for SPECT.
        psf : element in :class:`~odl.discr.lp_discr.DiscreteLp`
            Point Spread Functions  for the correction of the
            collimator-detector response
        use_gpu : `bool`, optional
            'False' if not using GPU
            `True` if using GPU provided that `impl` supports it
        impl='niftyrec', optional
            Implementation back-end for the transform. Supported back-ends:
            NiftyRec Tomography Toolbox through NiftyPy a Python wrapper
            for NiftyRec
        """
        impl, impl_in = str(impl).lower(), impl
        if impl not in _SUPPORTED_IMPL:
            raise ValueError('implementation {!r} not supported.'
                             ''.format(impl_in))

        self.geometry = geometry
        det_nx = geometry.det_grid.shape[0]
        det_ny = geometry.det_grid.shape[1]
        startAngle = float(geometry.motion_grid.points()[0])
        stopAngle = float(geometry.motion_grid.points()[-1]) * (180 / np.pi)
        N_proj = len(geometry.motion_grid.points())

        self._cameras = np.linspace(startAngle, stopAngle, N_proj,
                                    dtype=np.float32).reshape((N_proj, 1))
        self._use_gpu = use_gpu
        self._attenuation = attenuation
        self._psf = psf

        lowleft_x = geometry.detector.params.corners()[0][0]
        lowleft_y = geometry.detector.params.corners()[0][1]
        upright_x = geometry.detector.params.corners()[-1][0]
        upright_y = geometry.detector.params.corners()[-1][1]
        dom = odl.uniform_discr([lowleft_x, lowleft_y, startAngle],
                                [upright_x, upright_y, stopAngle],
                                [det_nx, det_ny, N_proj])

        super().__init__(dom, ran, linear=True)

    def _call(self, x):
        """Compute the discrete
        Parameters
        ----------
        x : `DiscreteLpVector`
           Element in the domain of the operator which is back-projected

        Returns
        -------
        out : `DiscreteLpVector`
            Returns an element in the projection space
        """
        if self._attenuation is None:
            attenuation = None
        else:
            attenuation = np.float32(
                np.asfortranarray(self._attenuation.asarray()))

        if self._psf is None:
            psf = None
        else:
            psf = np.float32(self._psf.asarray())

        x = np.float32(np.asfortranarray(x.asarray()))
        activity = SPECT_backproject_parallelholes(x, self._cameras,
                                                   attenuation, psf,
                                                   use_gpu=self._use_gpu)
        return self.range.element(activity)

    @property
    def adjoint(self):
        return SpectProject(dom=self.range,
                            geometry=self.geometry,
                            attenuation=self._attenuation,
                            psf=self._psf,
                            use_gpu=self._use_gpu)
