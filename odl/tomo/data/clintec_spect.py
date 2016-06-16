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
Houndsfield units see [Bro+2008]_ and references therein, for example.

"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import dicom
import os


def read_clintec_CT_reconstruction(path):
    """Reads the data in DICOM format and returns a HU CT volume.

    Parameters
    ----------
    path : `str`
        a path to the folder where DICOM data is

    Returns
    -------
    ct_volume : `numpy.ndarray`
        an array of CT reconstruction in Houndsfield units.
    dataset : `DICOM` File Meta Information

    """

    dcm_file_list = []
    for dirName, subdirList, fileList in os.walk(path):
        for filename in sorted(fileList):
            dcm_file_list.append(os.path.join(dirName, filename))

    dataset = dicom.read_file(dcm_file_list[0])

    shape = (int(dataset.Rows), int(dataset.Columns), len(dcm_file_list))
    data_array = np.zeros(shape, dtype=dataset.pixel_array.dtype)

    for i, fname in enumerate(dcm_file_list):
        ds = dicom.read_file(fname)
        data_array[:, :, i] = ds.pixel_array

    rescale_intercept = int(dataset.RescaleIntercept)
    rescale_slope = int(dataset.RescaleSlope)

    ct_volume = rescale_slope * np.float32(data_array) + rescale_intercept

    return ct_volume, dataset


def linear_attenuation_map_from_HU(volume, a, b):
    """Converts Houndsfield units to a linear attenuation map.

    Parameters
    ----------
    volume : `numpy.ndarray`
        an array of Houndsfield units to be converted
    a, b : `sequence`
        a = (a1, a2) where a1 `float` and a2 `float` are the intercepts
            of the bilinear curve
        b = (b1, b2) where b1 `float` and b2 `float` are the slopes
            of the bilinear curve
        see `Notes` for furher information

    Notes
    -----
        a and b are parameters of a bilinear curve for converting
        Houndsfield units to linear attenuation map, such that
        mu = a1 + b1 * ct_volume for HU < 0
        mu = a2 + b2 * ct_volume for HU >= 0

    Returns
    -------
    attenuation_map : `numpy.ndarray`
        An array of linear attenuation values.
    """
    a1, a2 = a
    b1, b2 = b
    mask = np.less(volume, 0)
    attenuation_map = np.empty_like(volume)
    attenuation_map[mask] = a1 + b1 * volume[mask]
    attenuation_map[~mask] = a2 + b2 * volume[~mask]
    attenuation_map = np.absolute(attenuation_map)
    return attenuation_map


def read_clintec_raw_spect_data(path):
    """Reads the raw spect DICOM data and retuns an array of the data.

    Parameters
    ----------
    path : `str`
        a path to the data folder

    Returns
    -------
    spect_data : `numpy.ndarray`
        an array of SPECT projection data
    dataset : `DICOM` File Meta Information
    """

    filenames = list(os.listdir(path))
    data_dicom = os.path.join(path, filenames[0])
    dataset = dicom.read_file(data_dicom)
    spect_data = dataset.pixel_array

    return spect_data, dataset
