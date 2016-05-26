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

"""Example reconstruction with stir.
In a fashion that would be performed by an ODL user.
Please have a look at the stir_reconstruction_2 for a more
STIR-like approach. """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


import os.path as pth
import odl

# Temporal edit to account for the stuff.

# Set path to input files
base = pth.join(pth.dirname(pth.abspath(__file__)), 'data', 'stir')

# volume_file = str(pth.join(base, 'initial.hv'))
#
# N.E. Replace the call to this function by creating a new ODL space and
# transform it to STIR domain.
#
# New ODL domain
# N.E. At a later point we are going to define a scanner with ring spacing 4.16
# therefore the z voxel size must be a divisor of that size.
discr_dom_odl = odl.tomo.stir_get_ODL_domain_which_honours_STIR_restrictions([151, 151, 151], [2.5, 2.5, 2.08])

stir_domain = odl.tomo.stir_get_STIR_domain_from_ODL(discr_dom_odl, 0.0)

#
#
# Now, let us create a ODL geometry and made a Scanner, and ProjData out of it.
#projection_file = str(pth.join(base, 'small.hs'))
#
# Instead of calling a hs file we are going to initialise a projector, based on a scanner,

#
# This would correspond to the mCT scanner
#
# Detector x size in mm - plus the ring difference
det_nx_mm = 4.16
# Detector y size in mm - plus the ring difference
det_ny_mm = 4.16
# Total number of rings
num_rings = 52
# Total number of detectors per ring
num_dets_per_ring = 624
# Inner radius of the scanner (crystal surface)
det_radius = 42.4 # in mm

#
# Additional things that STIR would like to know
#
average_depth_of_inter = 1.2 # in mm
ring_spacing = det_ny_mm
voxel_size_xy = 2.5 # in mm
axial_crystals_per_block = 13
trans_crystals_per_block = 13
axials_blocks_per_bucket = 4
trans_blocks_per_bucket_v = 1
axial_crystals_per_singles_unit = 13
trans_crystals_per_singles_unit = 13
num_detector_layers = 1
intrinsic_tilt = 0.0

# Create a PET geometry (ODL object) which is similar
# to the one that STIR will create using these values
geom = odl.tomo.stir_get_ODL_geometry_which_honours_STIR_restrictions(det_nx_mm, det_ny_mm,\
                                                                      num_rings, num_dets_per_ring,\
                                                                      det_radius)

# Now create the STIR geometry
stir_scanner = odl.tomo.stir_get_STIR_geometry(num_rings, num_dets_per_ring,\
                                               det_radius, ring_spacing,\
                                               average_depth_of_inter,\
                                               voxel_size_xy,\
                                               axial_crystals_per_block, trans_crystals_per_block,\
                                               axials_blocks_per_bucket, trans_blocks_per_bucket_v,\
                                               axial_crystals_per_singles_unit, trans_crystals_per_singles_unit,\
                                               num_detector_layers, intrinsic_tilt)



## Create a STIR projector from file data.
#proj = odl.tomo.backends.stir_bindings.stir_projector_from_file(
#    volume_file, projection_file)

#
#
# Parameters usefull to the projector setup
#

# Axial compression (Span)
# Reduction of the number of sinograms at different ring dierences
# as shown in STIR glossary.
# Span is a number used by CTI to say how much axial
# compression has been used.  It is always an odd number.
# Higher span, more axial compression.  Span 1 means no axial
# compression.
span_num = 1

# The segment is an index of the ring difference.
# In 2D PET there is only one segment = 0
# In 3D PET segment = 0 refers to direct sinograms
# The maximum number of segment can be 2*NUM_RINGS - 1
# Setting the followin variable to -1 implies : maximum possible
max_num_segments = -1

# If the views is less than half the number of detectors defined in
#  the Scanner then we subsample the scanner angular positions.
# If it is larger we are going to have empty cells in the sinogram
num_of_views = num_dets_per_ring / 2

# The number of tangestial positions refers to the last sinogram
# coordinate which is going to be the LOS's distance from the center
# of the FOV. Normally this would be the number of default_non_arc_bins
num_non_arccor_bins = num_dets_per_ring / 2

# A boolean if the data have been arccorrected during acquisition
# or in preprocessing. Anyways, STIR will not do that for you, but needs
# to know.
data_arc_corrected = False


# Now lets create the proper projector info
proj_info = odl.tomo.stir_get_projection_data_info(stir_domain,
                                                   stir_scanner, span_num,
                                                   max_num_segments, num_of_views,
                                                   num_non_arccor_bins, data_arc_corrected)

#
#Now lets create the projector data space (range)
#
initialize_to_zero = True

proj_data = odl.tomo.stir_get_projection_data(proj_info, initialize_to_zero)


proj = odl.tomo.backends.stir_bindings.stir_projector_from_memory(discr_dom_odl,\
                                                                  stir_domain,\
                                                                  proj_data)


# A sample phantom in odl domain
odl_phantom = odl.util.shepp_logan(discr_dom_odl, modified=True)

# A sample phantom to project
stir_phantom = odl.tomo.stir_get_STIR_image_from_ODL_Vector(discr_dom_odl, odl_phantom)

# Project data
projections = proj(odl_phantom)

# Calculate operator norm for landweber
op_norm_est_squared = proj.adjoint(projections).norm() / odl_phantom.norm()
omega = 0.5 / op_norm_est_squared

# Reconstruct using ODL
recon = proj.domain.zero()
odl.solvers.landweber(proj, recon, projections, niter=50, omega=omega)
recon.show()
