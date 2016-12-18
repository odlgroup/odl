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

"""Default functionals defined on any space similar to R^n or L^2."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
from numbers import Integral

from odl.operator import Operator
from odl.solvers.functional.functional import Functional
from skimage.restoration import denoise_nl_means
import cv2

__all__ = ('NLMRegularizer',)


class NLMRegularizer(Functional):

    """The nonlocal means functional.

    TODO: What is the actual functional?
    """

    def __init__(self, space, h,
                 patch_size=7, patch_distance=11, impl='skimage'):
        """Initialize a new instance.
        """
        self.h = float(h)
        self.impl = impl
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        super().__init__(space=space, linear=False, grad_lipschitz=np.nan)

    @property
    def proximal(self):
        func = self

        class NLMProximal(Operator):
            def __init__(self, stepsize):
                Operator.__init__(self, func.domain, func.domain, False)
                self.stepsize = stepsize

            def _call(self, x):
                h = func.h * self.stepsize

                if func.impl == 'skimage':
                    x_arr = x.asarray()
                    return denoise_nl_means(
                        x_arr,
                        patch_size=func.patch_size,
                        patch_distance=func.patch_distance,
                        h=h)
                elif func.impl == 'opencv':
                    x_arr = x.asarray()
                    xmin, xmax = np.min(x_arr), np.max(x_arr)
                    x_arr = (x_arr - xmin) * 255.0 / (xmax - xmin)
                    x_arr = x_arr.astype('uint8')

                    h_scaled = h * 255.0 / (xmax - xmin)
                    res = cv2.fastNlMeansDenoising(
                        x_arr,
                        templateWindowSize=func.patch_size,
                        searchWindowSize=2 * func.patch_distance + 1,
                        h=h_scaled)

                    return res * (xmax - xmin) / 255.0 + xmin
        return NLMProximal


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
