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

"""Example of reconstructing with CLINTEC data."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

import odl
from odl.tomo.operators.spect_trafo import AttenuatedRayTransform
from odl.tomo.data.clintec_spect import (read_clintec_raw_spect_data,
                                         read_clintec_CT_reconstruction,
                                         spect_clintec_geometries_from_file,
                                         linear_attenuation_from_HU)


# Put here your reference reconstruction
spect_reference = ''
spect_reco_reference, header_Hermes = read_clintec_raw_spect_data(
    spect_reference)

# Put your SPECT data here.
# This assumes that the data is in DICOM format
spect_file = ''

# Create SPECT geometry and the reconstruction domain from the spect-data file
spect_data, geometry, spect_domain = spect_clintec_geometries_from_file(
    spect_file)

# Create the SPECT projector without attenuation
A_no_attenuation = AttenuatedRayTransform(
    spect_domain, geometry, attenuation=None, impl='niftyrec_cpu')

# Compute first reconstruction without corrections
data = A_no_attenuation.range.element(spect_data)
# Create an initial starting point
x = A_no_attenuation.domain.one()
# Reconstruct using MLEM algorithm with few number of iterations
recon_noattenuation = odl.solvers.mlem(
    A_no_attenuation, x, data, iter=10, partial=None)


# Put the directory of your CT reconstructions here.
# The following assumes that CT is reconstructed as slices and
# that each slide is stored in a DICOM file
ct_path = ''
ct_hu, ct_header = read_clintec_CT_reconstruction(ct_path)

# Coefficients to convert Houndsfield units to linear
# attenuation map for SPECT
a = (0.152, 0.150)
b = (0.155 * 10 ** -3, 0.115 * 10 ** -3)
# Convert from Houndsfields to linear attenuation map
mu_map = linear_attenuation_from_HU(ct_hu, a, b)

# Permutation order for NiftyRec
mu_map = np.transpose(mu_map, (0, 2, 1))
mu_map = mu_map[:, ::-1, :]

# Co-registration step: make sure that the ct_hu and spect are aligned
# do co-registration if needed with the help of recon_noattenuation

# Resampling step: do resampling of linear attenuation volume if
# the size of mu_map is not that of recon_noattenuation
shape_mu_map = np.shape(mu_map)
begin = list(spect_domain.partition.begin)
end = list(spect_domain.partition.end)
domain_ct = odl.uniform_discr(begin, end,
                              [shape_mu_map[0], shape_mu_map[1],
                               shape_mu_map[2]], interp='linear')
resampling = odl.Resampling(domain_ct, spect_domain)
attenuation = resampling(mu_map)

# Create the SPECT projector with attenuation
A_attenuation = AttenuatedRayTransform(
    spect_domain, geometry, attenuation=attenuation, impl='niftyrec_cpu')

# Cast the array-like spect_data as element in the range of the operator
data = A_attenuation.range.element(spect_data)
# Create a nonnegative initial starting point
x = A_attenuation.domain.one()

# Reconstruct using MLEM
recon_attenuation = odl.solvers.mlem(
    A_attenuation, x, data, iter=50, partial=None)

# Display results
attenuation.show(indices=np.s_[:, 47, :], title='Attenuation map')
recon_noattenuation.show(indices=np.s_[:, 47, :], title='ODL reconstruction')
recon_attenuation.show(indices=np.s_[:, 47, :],
                       title='Attenuation corrected ODL reconstruction')
