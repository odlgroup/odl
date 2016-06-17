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
try:
    import dicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

import os
import re

__all__ = ('read_clintec_CT_reconstruction', 'linear_attenuation_from_HU',
           'read_clintec_raw_spect_data', 'DICOM_AVAILABLE')


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def read_clintec_CT_reconstruction(path):
    """Read DICOM format data to array.

    Parameters
    ----------
    path : `str`
        Path of the folder where DICOM data is stored.

    Returns
    -------
    ct_volume : `array-like`
        Array of CT reconstruction in Houndsfield units.

    dataset : DICOM File Meta Information

    """

    dcm_file_list = []
    for dir_name, _, file_list in os.walk(path):
        for filename in sorted(file_list, key=natural_sort_key):
            dcm_file_list.append(os.path.join(dir_name, filename))

    dataset = dicom.read_file(dcm_file_list[0])

    shape = (int(dataset.Rows), int(dataset.Columns), len(dcm_file_list))
    data_array = np.empty(shape, dtype=dataset.pixel_array.dtype)

    for i, fname in enumerate(dcm_file_list):
        ds = dicom.read_file(fname)
        data_array[:, :, i] = ds.pixel_array

    rescale_intercept = float(dataset.RescaleIntercept)
    rescale_slope = float(dataset.RescaleSlope)

    ct_volume = rescale_slope * data_array + rescale_intercept

    return ct_volume, dataset


def linear_attenuation_from_HU(volume, a, b):
    """Convert Houndsfield units to linear attenuation coefficient.

    Parameters
    ----------
    volume : `array-like`
        array of Houndsfield units to be converted

    a, b : `sequence`
        a = (a1, a2) where a1 `float` and a2 `float` are the intercepts
            of the bilinear curve
        b = (b1, b2) where b1 `float` and b2 `float` are the slopes
            of the bilinear curve
        see Notes_ for furher information.

    Returns
    -------
    attenuation : `array-like`
        array of linear attenuation values.

    Notes
    -----
        `a` and `b` are parameters of a bilinear curve for converting
        Houndsfield units (HU) to linear attenuation coefficients, such that

        mu = a1 + b1 * ct_volume for HU < 0

        mu = a2 + b2 * ct_volume for HU >= 0

        See [Bro+2008]_ and [Bur+2002]_ for example.

    """
    a1, a2 = a
    b1, b2 = b
    mask = np.less(volume, 0)
    attenuation = np.empty_like(volume)
    attenuation[mask] = a1 + b1 * volume[mask]
    attenuation[~mask] = a2 + b2 * volume[~mask]
    attenuation = np.absolute(attenuation)
    return attenuation


def read_clintec_raw_spect_data(file_dicom):
    """Read raw SPECT DICOM data to array.

    Parameters
    ----------
    path : `str`
        Name of the file where DICOM data is stored.

    Returns
    -------
    spect_data : `array-like`
        SPECT projection data
    dataset : DICOM File Meta Information
    """

    dataset = dicom.read_file(file_dicom)
    spect_data = dataset.pixel_array

    return spect_data, dataset
