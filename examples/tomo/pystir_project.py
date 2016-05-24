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

"""Example projection and back-projection with stir."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import os.path as pth

import odl

try:
    import pylab
    HAVE_PYLAB = True
except ImportError:
    HAVE_PYLAB = False


SIMPLE_GEOMETRY_CHECK = False
CHECK_IN_3D = False

SIMPLE_PHANTOM_CHECK = True

# Load STIR input files with data
base = pth.join(pth.join(pth.dirname(pth.abspath(__file__)), 'data'), 'stir')


if SIMPLE_GEOMETRY_CHECK:
    # --- Set-up empty geometry --- #

    #
    #
    # Discretised reconstruction space: discretized functions on the rectangle
    #

    if not CHECK_IN_3D:
        reco_space = odl.uniform_discr(
            min_corner=[-20, -20], max_corner=[20, 20], nsamples=[300, 300],
            dtype='float32')
    else:
        reco_space = odl.uniform_discr(
            min_corner=[-20, -20, -20], max_corner=[20, 20, 20], nsamples=[300, 300, 300],
            dtype='float32')
    #
    #
    # Create an empty image based on the reco_space size and discretisation in 3D
    #

    empty_image = odl.tomo.pstir_copy_DiscreteLP(reco_space)

    #
    #
    # DISPLAY
    #

    if HAVE_PYLAB:
        image = empty_image
        data = image.as_array()
        pylab.imshow(data[0,:,:])
        pylab.show()

if SIMPLE_PHANTOM_CHECK:

    #
    #
    # Discretised reconstruction space: discretized functions on the rectangle
    #
    reco_space = odl.uniform_discr(
            min_corner=[-20, -20, -20], max_corner=[20, 20, 20], nsamples=[151, 151, 151],
            dtype='float32')

    phantom = odl.util.shepp_logan(reco_space, modified=True)
    # phantom.show()

    stir_phantom = odl.tomo.pstir_copy_DiscreteLPVector(reco_space, phantom)

# empty_image_3D = odl.tomo.pstir_empty_volume_geometry(reco_space_3D)
# empty_image_3D.show()

# --- Copy geometry --- #

# Create shepp-logan phantom
# vol = odl.util.shepp_logan(reco_space, modified=True)
# vol.show()

# volume_file = str(pth.join(base, 'initial.hv'))
# volume = stir.FloatVoxelsOnCartesianGrid.read_from_file(volume_file)

# --- Save geometry --- #


# --- Retrieve geometry --- #


# --- Compare geometry --- #




# projection_file = str(pth.join(base, 'small.hs'))
# proj_data_in = stir.ProjData.read_from_file(projection_file)
# proj_data = stir.ProjDataInMemory(proj_data_in.get_exam_info(),
#                                   proj_data_in.get_proj_data_info())

# Create ODL spaces
# recon_sp = odl.uniform_discr([0, 0, 0], [1, 1, 1], (15, 64, 64))
# data_sp = odl.uniform_discr([0, 0, 0], [1, 1, 1], (37, 28, 56))

# Make STIR projector
# proj = odl.tomo.backends.stir_bindings.ForwardProjectorByBinWrapper(
#     recon_sp, data_sp, volume, proj_data)



# Project and show
#result = proj(vol)
#result.show()

# Also show back-projection
#back_projected = proj.adjoint(result)
#back_projected.show()
