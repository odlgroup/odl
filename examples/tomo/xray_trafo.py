# Copyright 2014, 2015 The ODL development group
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

"""Example using the discrete X-ray transform operator."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import numpy as np

# Internal
import odl


def parallel_2d():
    # Discrete reconstruction space
    discr_reco_space = odl.uniform_discr([-20, -20],
                                         [20, 20],
                                         [300, 300], dtype='float32')

    # Geometry
    agrid = odl.uniform_sampling(0, 2 * np.pi, 360)
    dgrid = odl.uniform_sampling(-30, 30, 558)
    geom = odl.tomo.Parallel2dGeometry(agrid, dgrid)

    # X-ray transform
    xray_trafo = odl.tomo.XrayTransform(discr_reco_space, geom,
                                        backend='astra_cuda')

    # Domain element
    discr_vol_data = odl.util.phantom.shepp_logan(discr_reco_space, True)

    # Forward projection
    discr_proj_data = xray_trafo(discr_vol_data)

    # Back projection
    discr_reco_data = xray_trafo.adjoint(discr_proj_data)

    # Shows a slice of the phantom, projections, and reconstruction
    discr_vol_data.show(title='parallel 2d volume')
    discr_proj_data.show(title='parallel 2d sinogram')
    discr_reco_data.show(title='parallel 2d backprojection')


def parallel_3d():
    # Discrete reconstruction space
    discr_reco_space = odl.uniform_discr([-20, -20, -20],
                                         [20, 20, 20],
                                         [300, 300, 300], dtype='float32')

    # Geometry
    agrid = odl.uniform_sampling(0, 2 * np.pi, 360)
    dgrid = odl.uniform_sampling([-30, -30], [30, 30], [558, 558])

    # Astra cannot handle axis aligned origin_to_det unless it is aligned
    # with the third coordinate axis. See issue #18 at ASTRA's github.
    # This is fixed in new versions of astra, with older versions, this could
    # give a zero result.
    geom = odl.tomo.Parallel3dGeometry(agrid, dgrid)

    # X-ray transform
    xray_trafo = odl.tomo.XrayTransform(discr_reco_space, geom,
                                        backend='astra_cuda')

    # Domain element
    discr_vol_data = odl.util.phantom.shepp_logan(discr_reco_space, True)

    # Forward projection
    discr_proj_data = xray_trafo(discr_vol_data)

    # Back projection
    discr_reco_data = xray_trafo.adjoint(discr_proj_data)

    # Shows a slice of the phantom, projections, and reconstruction
    discr_vol_data.show(indices=np.s_[:, :, 150],
                        title='parallel 3d volume')
    discr_proj_data.show(indices=np.s_[0, :, :],
                         title='parallel 3d projection 0')
    discr_reco_data.show(indices=np.s_[:, :, 150],
                         title='parallel 3d backprojection')


def fanbeam():
    # Discrete reconstruction space
    discr_reco_space = odl.uniform_discr([-20, -20],
                                         [20, 20],
                                         [300, 300], dtype='float32')

    # Geometry
    agrid = odl.uniform_sampling(0, 2 * np.pi, 360)
    dgrid = odl.uniform_sampling(-30, 30, 558)
    geom = odl.tomo.FanFlatGeometry(agrid, dgrid,
                                    src_radius=1000, det_radius=100)

    # X-ray transform
    xray_trafo = odl.tomo.XrayTransform(discr_reco_space, geom,
                                        backend='astra_cuda')

    # Domain element
    discr_vol_data = odl.util.phantom.shepp_logan(discr_reco_space, True)

    # Forward projection
    discr_proj_data = xray_trafo(discr_vol_data)

    # Back projection
    discr_reco_data = xray_trafo.adjoint(discr_proj_data)

    # Shows a slice of the phantom, projections, and reconstruction
    discr_vol_data.show(title='fan volume')
    discr_proj_data.show(title='fan sinogram')
    discr_reco_data.show(title='fan backprojection')


def conebeam():
    # Discrete reconstruction space
    discr_reco_space = odl.uniform_discr([-20, -20, -20],
                                         [20, 20, 20],
                                         [300, 300, 300], dtype='float32')

    # Geometry
    agrid = odl.uniform_sampling(0, 2 * np.pi, 360)
    dgrid = odl.uniform_sampling([-30, -30], [30, 30], [558, 558])
    geom = odl.tomo.CircularConeFlatGeometry(agrid, dgrid,
                                             src_radius=1000, det_radius=100,
                                             axis=[1, 0, 0])

    # X-ray transform
    xray_trafo = odl.tomo.XrayTransform(discr_reco_space, geom,
                                        backend='astra_cuda')

    # Domain element
    discr_vol_data = odl.util.phantom.shepp_logan(discr_reco_space, True)

    # Forward projection
    discr_proj_data = xray_trafo(discr_vol_data)

    # Back projection
    discr_reco_data = xray_trafo.adjoint(discr_proj_data)

    # Shows a slice of the phantom, projections, and reconstruction
    discr_vol_data.show(indices=np.s_[:, :, 150], title='cone volume')
    discr_proj_data.show(indices=np.s_[0, :, :], title='cone projection 0')
    discr_reco_data.show(indices=np.s_[:, :, 150], title='cone backprojection')


def helical():
    # Discrete reconstruction space
    discr_reco_space = odl.uniform_discr([-20, -20, 0],
                                         [20, 20, 40],
                                         [300, 300, 300], dtype='float32')

    # Geometry
    n_angle = 2000
    agrid = odl.uniform_sampling(0, 8 * 2 * np.pi, n_angle)
    dgrid = odl.uniform_sampling([-30, -3], [30, 3], [558, 60])
    geom = odl.tomo.HelicalConeFlatGeometry(agrid, dgrid, pitch=5.0,
                                            src_radius=1000, det_radius=100)

    # X-ray transform
    xray_trafo = odl.tomo.XrayTransform(discr_reco_space, geom,
                                        backend='astra_cuda')

    # Domain element
    discr_vol_data = odl.util.phantom.shepp_logan(discr_reco_space, True)

    # Forward projection
    discr_proj_data = xray_trafo(discr_vol_data)

    # Back projection
    discr_reco_data = xray_trafo.adjoint(discr_proj_data)

    # Shows a slice of the phantom, projections, and reconstruction
    discr_vol_data.show(indices=np.s_[150, :, :], title='cone volume x slice')
    discr_reco_data.show(indices=np.s_[150, :, :],
                         title='cone backprojection x')
    discr_proj_data.show(indices=np.s_[1000, :, :],
                         title='cone projection 1000')


if __name__ == '__main__':
    parallel_2d()
    parallel_3d()
    fanbeam()
    conebeam()
    helical()
