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

"""Helpers for reading CLINTEC SPECT data formats.

For the bilinear relationship of linear attenuation coefficients and
Houndsfield units see, for example

Investigation of the relationship between linear attenuation
coefficients and CT Houndsfield units using radionuclides for SPECT
S. Brown et al, Applied Radiation and Isotopes 66 (2008) 1206-1212

and references therein.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import dicom
import os
from natsort import natsorted


def read_clintec_CT_reconstruction(path):
    """Reads the data in Dicom format and returns a HU CT volume.

    Parameters
    ----------
    path : a `path` to the folder where Dicom data is

    """

    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(path):
        for filename in natsorted(fileList):
            if ".dcm" in filename.lower():
                # check whether the file ends by .dcm
                lstFilesDCM.append(os.path.join(dirName, filename))
            else:
                lstFilesDCM.append(os.path.join(dirName, filename))

    dataset = dicom.read_file(lstFilesDCM[0])

    # Load dimensions based on the number of rows, columns, and slices
    shape = (int(dataset.Rows), int(dataset.Columns), len(lstFilesDCM))
    data_array = np.zeros(shape, dtype=dataset.pixel_array.dtype)
    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = dicom.read_file(filenameDCM)
        # store the raw image data
        data_array[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    rescale_intercept = int(dataset.RescaleIntercept)
    rescale_slope = int(dataset.RescaleSlope)

    ct_volume = rescale_slope*np.float32(data_array) + rescale_intercept

    return ct_volume, dataset


def linear_attenuation_map_from_HU_data(path, a, b):
    """Converts HU data to linear attenuation map.

    Parameters
    ----------
    path : a `path` to the data folder
    a, b : `tuple`
        a = (a1, a2) where a1 `float` and a2 `float` are the intercepts
            of the bilinear curve
        b = (b1, b2) where b1 `float` and b2 `float` are the slopes
            of the bilinear curve
        a and b are parameters of a bilinear curve for converting
        HU-values to linear attenuation map, such that
        mu = a1 + b1 * ct_volume for HU < 0
        mu = a2 + b2 * ct_volume for HU >= 0

    """
    a1, a2 = a
    b1, b2 = b

    ct_volume, dataset = read_clintec_CT_reconstruction(path)
    (i1, j1, k1) = np.where(ct_volume < 0)
    (i2, j2, k2) = np.where(ct_volume >= 0)

    attenuation_map = np.zeros_like(ct_volume)
    attenuation_map = np.zeros_like(ct_volume)
    attenuation_map[i1, j1, k1] = a1 + b1 * ct_volume[i1, j1, k1]
    attenuation_map[i2, j2, k2] = a2 + b2 * ct_volume[i2, j2, k2]
    attenuation_map = np.absolute(attenuation_map)

    return attenuation_map


def read_clintec_raw_spect_data(path):
    """Reads the raw spect Dicom data and retuns an array of the data."""

    filenames = [f for f in os.listdir(path)]
    data_dicom = os.path.join(path, filenames[0])
    dataset = dicom.read_file(data_dicom)
    spect_data = np.float32(dataset.pixel_array)

    return spect_data
