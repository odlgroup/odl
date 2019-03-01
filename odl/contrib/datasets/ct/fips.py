# Copyright 2014-2017 The ODL contributors
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

"""Tomographic datasets from the Finish Inverse Problems Society (FIPS)."""

from __future__ import division
import numpy as np
from odl.contrib.datasets.util import get_data
from odl.discr import uniform_partition
from odl.tomo import FanBeamGeometry


__all__ = ('walnut_data', 'walnut_geometry',
           'lotus_root_data', 'lotus_root_geometry')


DATA_SUBSET = 'ct_fips'


def walnut_data():
    """Tomographic X-ray data of a walnut.

    Notes
    -----
    See the article `Tomographic X-ray data of a walnut`_ for further
    information.

    See Also
    --------
    walnut_geometry

    References
    ----------
    .. _Tomographic X-ray data of a walnut: https://arxiv.org/abs/1502.04064
    """
    # TODO: Store data in some ODL controlled url
    url = 'http://www.fips.fi/dataset/CT_walnut_v1/FullSizeSinograms.mat'
    dct = get_data('walnut.mat', subset=DATA_SUBSET, url=url)

    # Change axes to match ODL definitions
    data = np.swapaxes(dct['sinogram1200'], 0, 1)[::-1, ::-1]
    data = data.astype('float')

    # Very crude gain normalization
    data = -np.log(data / np.max(data, axis=1)[:, None])
    return data


def walnut_geometry():
    """Tomographic geometry for the walnut dataset.

    Notes
    -----
    See the article `Tomographic X-ray data of a walnut`_ for further
    information.

    See Also
    --------
    walnut_data

    References
    ----------
    .. _Tomographic X-ray data of a walnut: https://arxiv.org/abs/1502.04064
    """
    # To get the same rotation as in the reference article
    a_offset = -np.pi / 2
    apart = uniform_partition(a_offset, a_offset + 2 * np.pi, 1200)

    # TODO: Find exact value, determined experimentally
    d_offset = -0.279
    dpart = uniform_partition(d_offset - 57.4, d_offset + 57.4, 2296)

    geometry = FanBeamGeometry(apart, dpart,
                               src_radius=110, det_radius=190)

    return geometry


def lotus_root_data():
    """Tomographic X-ray data of a lotus root.

    Notes
    -----
    See the article `Tomographic X-ray data of a lotus root filled with
    attenuating objects`_ for further information.

    See Also
    --------
    lotus_root_geometry

    References
    ----------
    .. _Tomographic X-ray data of a lotus root filled with attenuating objects:
       https://arxiv.org/abs/1609.07299
    """
    # TODO: Store data in some ODL controlled url
    url = 'http://www.fips.fi/dataset/CT_Lotus_v1/sinogram.mat'
    dct = get_data('lotus_root.mat', subset=DATA_SUBSET, url=url)

    # Change axes to match ODL definitions
    data = np.swapaxes(dct['sinogram'], 0, 1)[:, :]
    data = data.astype('float')

    return data


def lotus_root_geometry():
    """Tomographic geometry for the lotus root dataset.

    Notes
    -----
    See the article `Tomographic X-ray data of a lotus root filled with
    attenuating objects`_ for further information.

    See Also
    --------
    lotus_root_geometry

    References
    ----------
    .. _Tomographic X-ray data of a lotus root filled with attenuating objects:
       https://arxiv.org/abs/1609.07299
    """
    # To get the same rotation as in the reference article
    a_offset = np.pi / 2
    apart = uniform_partition(a_offset,
                              a_offset + 2 * np.pi * 366. / 360.,
                              366)

    # TODO: Find exact value, determined experimentally
    d_offset = 0.35
    dpart = uniform_partition(d_offset - 60, d_offset + 60, 2240)

    geometry = FanBeamGeometry(apart, dpart,
                               src_radius=540, det_radius=90)

    return geometry


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
