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

"""Example using the ASTRA CUDA for 3D geometries."""

# pylint: disable=invalid-name,no-name-in-module

from __future__ import print_function, division, absolute_import
import os.path as pth
from future import standard_library

standard_library.install_aliases()

# External
import numpy as np
import pytest
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Internal
import odl


def save_ortho_slices(data, name, sli):
    """Save three orthogonal slices of the 3D input volume.

    Parameters
    ----------
    data : `DiscreteLp`
    name : `str`
    sli : 3-element array-like
    """
    # indices for the orthogonal slices
    x, y, z = np.asarray(sli, int)
    path = pth.join(pth.join(pth.dirname(pth.abspath(__file__)), 'temp'),
                    'astra')

    data.show('imshow',
              saveto=pth.join(path, '{}_z{:03d}.png'.format(
                      name.replace(' ', '_'), z)),
              title='{} [:,:,{}]'.format(name, z),
              indices=[slice(None), slice(None), z])
    data.show('imshow',
              saveto=pth.join(path, '{}_y{:03d}.png'.format(
                      name.replace(' ', '_'), y)),
              title='{} [:,{},:]'.format(name, y),
              indices=[slice(None), y, slice(None)])
    data.show('imshow',
              saveto=pth.join(path, '{}_x{:03d}.png'.format(
                      name.replace(' ', '_'), x)),
              title='{} [{},:,:]'.format(name, x),
              indices=[x, slice(None), slice(None)])

    plt.close('all')


def astra_cuda_parallel3d():
    # `DiscreteLp` volume space
    vol_shape = (80, 70, 60)
    discr_vol_space = odl.uniform_discr([-0.8, -0.7, -0.6], [0.8, 0.7, 0.6],
                                        vol_shape, dtype='float32')

    # Create an element in the volume space
    discr_data = odl.util.phantom.cuboid(discr_vol_space,
                                         (0.2, 0.4, 0.1,), (0.8, 0.4, 0.4))

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 110)

    # Detector
    dparams = odl.Rectangle([-1, -0.9], [1, 0.9])
    det_grid = odl.uniform_sampling(dparams, (100, 90))

    # Create geometry
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, angle_grid,
                                       det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # Indices of ortho slices
    vol_sli = np.round(0.25 * np.array(vol_shape))
    proj_sli = (0, np.round(0.25 * proj_shape[1]),
                np.round(0.25 * proj_shape[2]))
    save_ortho_slices(discr_data, 'phantom 3d cuda', vol_sli)

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector_call(discr_data, geom,
                                                           discr_proj_space)
    save_ortho_slices(proj_data, 'forward parallel 3d cuda', proj_sli)

    # Backward
    rec_data = odl.tomo.astra_cuda_backward_projector_call(proj_data, geom,
                                                           discr_vol_space)
    save_ortho_slices(rec_data, 'backward parallel 3d cuda', vol_sli)


def astra_cuda_conebeam_circular():
    # `DiscreteLp` volume space
    vol_shape = (80, 70, 60)
    discr_vol_space = odl.uniform_discr([-0.8, -0.7, -0.6], [0.8, 0.7, 0.6],
                                        vol_shape, dtype='float32')

    # Create an element in the volume space
    discr_data = odl.util.phantom.cuboid(discr_vol_space,
                                         (0.2, 0.4, 0.1,), (0.8, 0.4, 0.4))

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 110)

    # Detector
    dparams = odl.Rectangle([-1, -0.9], [1, 0.9])
    det_grid = odl.uniform_sampling(dparams, (100, 90))

    # Parameters
    src_rad = 1000
    det_rad = 100

    # Create geometries
    geom = odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams, src_rad,
                                             det_rad, angle_grid, det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # Indices of ortho slices
    vol_sli = np.round(0.25 * np.array(vol_shape))
    proj_sli = (0, np.round(0.25 * proj_shape[1]),
                np.round(0.25 * proj_shape[2]))
    save_ortho_slices(discr_data, 'phantom 3d cuda', vol_sli)

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector_call(discr_data, geom,
                                                           discr_proj_space)

    save_ortho_slices(proj_data, 'forward conebeam circular cuda', proj_sli)

    # Backward
    rec_data = odl.tomo.astra_cuda_backward_projector_call(proj_data, geom,
                                                           discr_vol_space)

    save_ortho_slices(rec_data, 'backward conebeam circular cuda', vol_sli)


def astra_cuda_conebeam_helical():
    # `DiscreteLp` volume space
    vol_shape = (80, 70, 60)
    discr_vol_space = odl.uniform_discr([-0.8, -0.7, -0.6], [0.8, 0.7, 0.6],
                                        vol_shape, dtype='float32')

    # Create an element in the volume space
    discr_data = odl.util.phantom.cuboid(discr_vol_space,
                                         (0.2, 0.4, 0.1,), (0.8, 0.4, 0.4))

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 110)

    # Detector
    dparams = odl.Rectangle([-1, -0.9], [1, 0.9])
    det_grid = odl.uniform_sampling(dparams, (100, 90))

    # Parameter
    src_rad = 1000
    det_rad = 100
    pitch_factor = 0.5

    # Create geometries
    geom = odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                            det_rad, pitch_factor, angle_grid,
                                            det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # Indices of ortho slices
    vol_sli = np.round(0.25 * np.array(vol_shape))
    proj_sli = (0, np.round(0.25 * proj_shape[1]),
                np.round(0.25 * proj_shape[2]))

    save_ortho_slices(discr_data, 'phantom 3d cuda', vol_sli)

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector_call(discr_data, geom,
                                                           discr_proj_space)

    save_ortho_slices(proj_data, 'forward conebeam circular cuda', proj_sli)

    # Backward
    rec_data = odl.tomo.astra_cuda_backward_projector_call(proj_data, geom,
                                                           discr_vol_space)

    save_ortho_slices(rec_data, 'backward conebeam circular cuda', vol_sli)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')