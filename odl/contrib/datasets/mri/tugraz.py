# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
#
# The data is licensed under a
# Creative Commons Attribution 4.0 International License.
#
# You should have received a copy of the license along with this
# work. If not, see <http://creativecommons.org/licenses/by/4.0/>.

"""MRI datasets provided by TU Graz."""

import numpy as np
from packaging.version import parse as parse_version

from odl.contrib.datasets.util import get_data
import odl

if parse_version(np.__version__) < parse_version('1.12'):
    flip = odl.util.npy_compat.flip
else:
    flip = np.flip


__all__ = ('mri_head_data_4_channel', 'mri_head_reco_op_4_channel',
           'mri_head_data_32_channel', 'mri_head_reco_op_32_channel',
           'mri_knee_data_8_channel', 'mri_knee_reco_op_8_channel')


DATA_SUBSET = 'trafos_tugraz'


def mri_head_data_4_channel():
    """Raw data for 4 channel MRI of a head.

    This is a T2 weighted TSE scan of a healthy volunteer.

    The data has been rescaled so that the reconstruction fits approximately
    in [0, 1].

    See the data source with DOI `10.5281/zenodo.800525`_ or the
    `project webpage`_ for further information.

    See Also
    --------
    mri_head_inverse_4_channel

    References
    ----------
    .. _10.5281/zenodo.800525: https://zenodo.org/record/800525
    .. _project webpage: http://imsc.uni-graz.at/mobis/internal/\
platform_aktuell.html
    """
    # TODO: Store data in some ODL controlled url
    url = 'https://zenodo.org/record/800525/files/1_rawdata_brainT2_4ch.mat'
    dct = get_data('1_rawdata_brainT2_4ch.mat', subset=DATA_SUBSET, url=url)

    # Change axes to match ODL definitions
    data = flip(np.swapaxes(dct['rawdata'], 0, -1) * 4e4, 2)

    return data


def mri_head_reco_op_4_channel():
    """Reconstruction operator for 4 channel MRI of a head.

    This is a T2 weighted TSE scan of a healthy volunteer.

    The reconstruction operator is the sum of the modulus of each channel.

    See the data source with DOI `10.5281/zenodo.800525`_ or the
    `project webpage`_ for further information.

    See Also
    --------
    mri_head_data_4_channel

    References
    ----------
    .. _10.5281/zenodo.800525: https://zenodo.org/record/800525
    .. _project webpage: http://imsc.uni-graz.at/mobis/internal/\
platform_aktuell.html
    """
    # To get the same rotation as in the reference article
    space = odl.uniform_discr(min_pt=[-115.2, -115.2],
                              max_pt=[115.2, 115.2],
                              shape=[256, 256],
                              dtype=complex)

    trafo = odl.trafos.FourierTransform(space)

    return odl.ReductionOperator(odl.ComplexModulus(space) * trafo.inverse, 4)


def mri_head_data_32_channel():
    """Raw data for 32 channel MRI of a head.

    This is a T2 weighted TSE scan of a healthy volunteer.

    The data has been rescaled so that the reconstruction fits approximately in
    [0, 1].

    See the data source with DOI `10.5281/zenodo.800527`_ or the
    `project webpage`_ for further information.

    See Also
    --------
    mri_head_inverse_32_channel

    References
    ----------
    .. _10.5281/zenodo.800527: https://zenodo.org/record/800527
    .. _project webpage: http://imsc.uni-graz.at/mobis/internal/\
platform_aktuell.html
    """
    # TODO: Store data in some ODL controlled url
    url = 'https://zenodo.org/record/800527/files/2_rawdata_brainT2_32ch.mat'
    dct = get_data('2_rawdata_brainT2_32ch.mat', subset=DATA_SUBSET, url=url)

    # Change axes to match ODL definitions
    data = flip(np.swapaxes(dct['rawdata'], 0, -1) * 7e3, 2)

    return data


def mri_head_reco_op_32_channel():
    """Reconstruction operator for 32 channel MRI of a head.

    This is a T2 weighted TSE scan of a healthy volunteer.

    The reconstruction operator is the sum of the modulus of each channel.

    See the data source with DOI `10.5281/zenodo.800527`_ or the
    `project webpage`_ for further information.

    See Also
    --------
    mri_head_data_32_channel

    References
    ----------
    .. _10.5281/zenodo.800529: https://zenodo.org/record/800527
    .. _project webpage: http://imsc.uni-graz.at/mobis/internal/\
platform_aktuell.html
    """
    # To get the same rotation as in the reference article
    space = odl.uniform_discr(min_pt=[-115.2, -115.2],
                              max_pt=[115.2, 115.2],
                              shape=[256, 256],
                              dtype=complex)

    trafo = odl.trafos.FourierTransform(space)

    return odl.ReductionOperator(odl.ComplexModulus(space) * trafo.inverse, 32)


def mri_knee_data_8_channel():
    """Raw data for 8 channel MRI of a knee.

    This is an SE measurement of the knee of a healthy volunteer.

    The data has been rescaled so that the reconstruction fits approximately in
    [0, 1].

    See the data source with DOI `10.5281/zenodo.800529`_ or the
    `project webpage`_ for further information.

    See Also
    --------
    mri_knee_inverse_8_channel

    References
    ----------
    .. _10.5281/zenodo.800529: https://zenodo.org/record/800529
    .. _project webpage: http://imsc.uni-graz.at/mobis/internal/\
platform_aktuell.html
    """
    # TODO: Store data in some ODL controlled url
    url = 'https://zenodo.org/record/800529/files/3_rawdata_knee_8ch.mat'
    dct = get_data('3_rawdata_knee_8ch.mat', subset=DATA_SUBSET, url=url)

    # Change axes to match ODL definitions
    data = flip(np.swapaxes(dct['rawdata'], 0, -1) * 9e3, 2)

    return data


def mri_knee_reco_op_8_channel():
    """Reconstruction operator for 8 channel MRI of a knee.

    This is an SE measurement of the knee of a healthy volunteer.

    The reconstruction operator is the sum of the modulus of each channel.

    See the data source with DOI `10.5281/zenodo.800529`_ or the
    `project webpage`_ for further information.

    See Also
    --------
    mri_knee_data_8_channel

    References
    ----------
    .. _10.5281/zenodo.800529: https://zenodo.org/record/800529
    .. _project webpage: http://imsc.uni-graz.at/mobis/internal/\
platform_aktuell.html
    """
    # To get the same rotation as in the reference article
    space = odl.uniform_discr(min_pt=[-74.88, -74.88],
                              max_pt=[74.88, 74.88],
                              shape=[192, 192],
                              dtype=complex)

    trafo = odl.trafos.FourierTransform(space)

    return odl.ReductionOperator(odl.ComplexModulus(space) * trafo.inverse, 8)
