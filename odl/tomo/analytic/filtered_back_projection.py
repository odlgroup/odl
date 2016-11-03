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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from odl.trafos import FourierTransform


__all__ = ('fbp_op',)


def fbp_op(ray_trafo):
    """Create Filtered BackProjection from a ray transform.

    Parameters
    ----------
    ray_trafo : `RayTransform`

    Returns
    -------
    fbp : `Operator`
        Approximate inverse operator of ``ray_trafo``.
    """
    if ray_trafo.domain.ndim == 2:
        fourier = FourierTransform(ray_trafo.range, axes=1)

        # Define ramp filter
        def fft_filter(x):
            return np.abs(x[1]) / (4 * np.pi)

    elif ray_trafo.domain.ndim == 3:
        fourier = FourierTransform(ray_trafo.range, axes=[1, 2])

        # Find the direction that the filter should be taken in
        src_to_det = ray_trafo.geometry.src_to_det_init
        axis = ray_trafo.geometry.axis
        rot_dir = np.cross(axis, src_to_det)
        c1 = np.vdot(rot_dir, ray_trafo.geometry.det_init_axes[0])
        c2 = np.vdot(rot_dir, ray_trafo.geometry.det_init_axes[1])

        # Define ramp filter
        def fft_filter(x):
            return np.sqrt(c1 * x[1] ** 2 + c2 * x[2] ** 2) / (2 * np.pi ** 2)
    else:
        raise NotImplementedError('FBP only implemented in 2d and 3d')

    # Create ramp in the detector direction
    ramp_function = fourier.range.element(fft_filter)

    # Create ramp filter via the
    # convolution formula with fourier transforms
    ramp_filter = fourier.inverse * ramp_function * fourier

    # Create filtered backprojection by composing the backprojection
    # (adjoint) with the ramp filter. Also apply a scaling.
    return ray_trafo.adjoint * ramp_filter


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
