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

"""Create image to check for the ASTRA cpu backend."""

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


def save_slice(data, name):
    """Save image.

    Parameters
    ----------
    data : `DiscreteLp`
    name : `str`
    """
    path = pth.join(pth.join(pth.dirname(pth.abspath(__file__)), 'temp'),
                    'astra')

    filename = '{}.png'.format(name.replace(' ', '_'))
    path = pth.join(path, filename)

    data.show('imshow', saveto=path, title='{} [:,:]'.format(name))

    plt.close()


def test_astra_cpu_parallel2d():

    # Create `DiscreteLp` space for volume data
    vol_shape = (100, 110)
    discr_vol_space = odl.uniform_discr([-1, -1.1], [1, 1.1], vol_shape,
                                        dtype='float32')

    # Create an element in the volume space
    discr_vol_data = odl.util.phantom.cuboid(discr_vol_space, 0.2, 0.3)

    save_slice(discr_vol_data, 'phantom 2d cpu')

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 180)

    # Detector
    dparams = odl.Interval(-2, 2)
    det_grid = odl.uniform_sampling(dparams, 100)

    # Create geometry instances
    geom = odl.tomo.Parallel2dGeometry(angle_intvl, dparams, angle_grid,
                                           det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, geom.grid.shape,
                                                   dtype='float32',
                                                   impl='numpy')

    # forward
    proj_data = odl.tomo.astra_cpu_forward_projector_call(discr_vol_data, geom,
                                                          discr_proj_space)
    save_slice(proj_data, 'forward parallel 2d cpu')

    # backward
    reco_data = odl.tomo.astra_cpu_backward_projector_call(proj_data, geom,
                                                           discr_vol_space)
    save_slice(reco_data, 'backward parallel 2d cpu')


def test_astra_cpu_fanflat():

    # Create `DiscreteLp` space for volume data
    vol_shape = (100, 110)
    discr_vol_space = odl.uniform_discr([-1, -1.1], [1, 1.1], vol_shape,
                                        dtype='float32')

    # Create an element in the volume space
    discr_vol_data = odl.util.phantom.cuboid(discr_vol_space, 0.2, 0.3)

    save_slice(discr_vol_data, 'phantom 2d cpu')

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 180)

    # Detector
    dparams = odl.Interval(-2, 2)
    det_grid = odl.uniform_sampling(dparams, 100)

    # Distance for fanflat geometries
    src_rad = 1000
    det_rad = 100

    # Create geometry instances
    geom = odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                       angle_grid, det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, geom.grid.shape,
                                                   dtype='float32',
                                                   impl='numpy')

    # Fanflat: forward
    discr_vol_data = odl.util.phantom.cuboid(discr_vol_space, 0.2, 0.3)
    proj_data = odl.tomo.astra_cpu_forward_projector_call(discr_vol_data,
                                                             geom,
                                                             discr_proj_space)
    save_slice(proj_data, 'forward fanflat cpu')

    # Fanflat: backward
    reco_data = odl.tomo.astra_cpu_backward_projector_call(proj_data, geom,
                                                           discr_vol_space)
    save_slice(reco_data, 'backward fanflat cpu')


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
