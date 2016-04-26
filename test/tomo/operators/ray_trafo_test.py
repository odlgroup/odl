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

"""Tests for the Ray transform."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import pytest
import numpy as np

# Internal
import odl
import odl.tomo as tomo
from odl.util.testutils import almost_equal


# Find the valid projectors
projectors = []
if tomo.ASTRA_AVAILABLE:
    projectors += ['par2d astra_cpu uniform',
                   'cone2d astra_cpu uniform'
                   # 'par2d astra_cpu random',
                   # 'cone2d astra_cpu random'
                   ]
if tomo.ASTRA_CUDA_AVAILABLE:
    projectors += ['par2d astra_cuda uniform',
                   'cone2d astra_cuda uniform',
                   'par3d astra_cuda uniform',
                   'cone3d astra_cuda uniform',
                   # 'cone3d cuda random',  # TODO: expected fail, fix issue!
                   'helical astra_cuda uniform'
                   ]
if tomo.SCIKIT_IMAGE_AVAILABLE:
    projectors += ['par2d scikit uniform']

projector_ids = [' {} '.format(p) for p in projectors]


@pytest.fixture(scope="module", params=projectors, ids=projector_ids)
def projector(request):
    n_voxels = 100
    n_angles = 100
    n_pixels = 100

    geom, impl, angle = request.param.split()

    if angle == 'uniform':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
    elif angle == 'random':
        min_pt = 2 * (2.0 * np.pi) / n_angles
        max_pt = (2.0 * np.pi) - 2 * (2.0 * np.pi) / n_angles
        points = np.sort(np.random.rand(n_angles)) * (max_pt - min_pt) + min_pt
        agrid = odl.TensorGrid(points)
        apart = odl.RectPartition(odl.Interval(min_pt, max_pt), agrid)
    else:
        raise ValueError('angle not valid')

    if geom == 'par2d':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n_voxels] * 2,
                                       dtype='float32')

        # Geometry
        dpart = odl.uniform_partition(-30, 30, n_pixels)
        geom = tomo.Parallel2dGeometry(apart, dpart)

        return tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'par3d':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n_voxels] * 3,
                                       dtype='float32')

        # Geometry
        dpart = odl.uniform_partition([-30] * 2, [30] * 2, [n_pixels] * 2)
        geom = tomo.Parallel3dAxisGeometry(apart, dpart)

        # Ray transform
        return tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone2d':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n_voxels] * 2,
                                       dtype='float32')

        # Geometry
        dpart = odl.uniform_partition(-30, 30, n_pixels)
        geom = tomo.FanFlatGeometry(apart, dpart, src_radius=200,
                                    det_radius=100)

        # Ray transform
        return tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone3d':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n_voxels] * 3,
                                       dtype='float32')

        # Geometry
        dpart = odl.uniform_partition([-30] * 2, [30] * 2, [n_pixels] * 2)

        geom = tomo.CircularConeFlatGeometry(apart, dpart, src_radius=200,
                                             det_radius=100)

        # Ray transform
        return tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'helical':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20, -20, 0], [20, 20, 40],
                                       [n_voxels] * 3, dtype='float32')

        # overwrite angle
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], [n_pixels] * 2)
        geom = tomo.HelicalConeFlatGeometry(apart, dpart, pitch=5.0,
                                            src_radius=200, det_radius=100)

        # Ray transform
        return tomo.RayTransform(reco_space, geom, impl=impl)
    else:
        raise ValueError('geom not valid')


def test_projector(projector):
    """Test discrete Ray transform using ASTRA for reconstruction."""

    # TODO: this needs to be improved
    # Accept 10% errors
    places = 1

    # Create shepp-logan phantom
    vol = projector.domain.one()

    # Calculate projection
    proj = projector(vol)

    # We expect maximum value to be along diagonal
    expected_max = projector.domain.partition.extent()[0] * np.sqrt(2)
    assert almost_equal(proj.ufunc.max(), expected_max, places=places)

    # Adjoint definition <Ax, Ax> = <x, A*A x>
    result_AxAx = proj.inner(proj)
    result_xAtAx = vol.inner(projector.adjoint(proj))
    assert almost_equal(result_AxAx, result_xAtAx, places=places)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
