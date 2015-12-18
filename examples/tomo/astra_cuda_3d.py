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
import matplotlib.pyplot as plt
import numpy as np

# Internal
from odl import (Interval, Rectangle, FunctionSpace, uniform_discr,
                 uniform_discr_fromspace, uniform_sampling,
                 Parallel3dGeometry, CircularConeFlatGeometry,
                 HelicalConeFlatGeometry, astra_gpu_forward_projector_call,
                 astra_gpu_backward_projector_call)


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
    path = pth.join(
        pth.join(pth.dirname(pth.abspath(__file__)), 'data'), 'astra')

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

# `DiscreteLp` volume space
vol_shape = (80, 70, 60)
discr_vol_space = uniform_discr([-0.8, -0.7, -0.6], [0.8, 0.7, 0.6],
                                vol_shape, dtype='float32')

# Phantom
phan = np.zeros(vol_shape)
sli0 = np.round(0.1 * np.array(vol_shape)).astype(int)
sli1 = np.round(0.4 * np.array(vol_shape)).astype(int)
sliz0 = np.round(0.1 * np.array(vol_shape)).astype(int)
sliz1 = np.round(1.9 * np.array(vol_shape)).astype(int)
phan[sliz0[0]:sliz1[0], sli0[1]:sli1[1], sli0[2]:sli1[2]] = 1

# Create an element in the volume space
discr_data = discr_vol_space.element(phan)

# Indices of ortho slices
vol_sli = np.round(0.25 * np.array(vol_shape))
save_ortho_slices(discr_data, 'phantom 3d gpu', vol_sli)

# Angles
angle_intvl = Interval(0, 2 * np.pi)
angle_grid = uniform_sampling(angle_intvl, 110, as_midp=False)

# Detector
dparams = Rectangle([-1, -0.9], [1, 0.9])
det_grid = uniform_sampling(dparams, (100, 90))

# Parameter for cone beam geometries
src_rad = 1000
det_rad = 100
spiral_pitch_factor = 0.5

# Create geometries
geom_p3d = Parallel3dGeometry(angle_intvl, dparams, angle_grid, det_grid)
geom_ccf = CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                    angle_grid, det_grid)
geom_hcf = HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                   det_rad, spiral_pitch_factor,
                                   angle_grid, det_grid)

# Projection space
proj_space = FunctionSpace(geom_p3d.params)

# `DiscreteLp` projection space
proj_shape = (angle_grid.ntotal, det_grid.shape[0], det_grid.shape[1])


discr_proj_space = uniform_discr_fromspace(proj_space, proj_shape,
                                           dtype='float32')

# Indices of ortho slices of projections to be saved
proj_sli = (0, np.round(0.25 * proj_shape[1]), np.round(0.25 * proj_shape[2]))

# Forward and back projections

# Forward: Parallel 3D
proj_data = astra_gpu_forward_projector_call(discr_data, geom_p3d,
                                             discr_proj_space)
save_ortho_slices(proj_data, 'forward parallel 3d gpu', proj_sli)

# Backward: Parallel 3D
rec_data = astra_gpu_backward_projector_call(proj_data, geom_p3d,
                                             discr_vol_space)
save_ortho_slices(rec_data, 'backward parallel 3d gpu', vol_sli)

# Forward: Circular Cone Flat
proj_data = astra_gpu_forward_projector_call(discr_data, geom_ccf,
                                             discr_proj_space)
save_ortho_slices(proj_data, 'forward conebeam circular gpu', proj_sli)

# Backward: Circular Cone Flat
rec_data = astra_gpu_backward_projector_call(proj_data, geom_ccf,
                                             discr_vol_space)
save_ortho_slices(rec_data, 'backward conebeam circular gpu', vol_sli)

# Forward: Helical Cone Flat
proj_data = astra_gpu_forward_projector_call(discr_data, geom_hcf,
                                             discr_proj_space)
save_ortho_slices(proj_data, 'forward conebeam helical gpu', proj_sli)

# Backward: Helical Cone Flat
rec_data = astra_gpu_backward_projector_call(proj_data, geom_hcf,
                                             discr_vol_space)
save_ortho_slices(rec_data, 'backward conebeam helical gpu', vol_sli)
