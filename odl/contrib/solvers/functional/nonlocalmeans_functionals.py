# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Non Local Means functionals."""

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.operator import Operator
from odl.solvers.functional.functional import Functional

__all__ = ('NLMRegularizer',)


class NLMRegularizer(Functional):

    r"""The nonlocal means "functional".

    This is not a true functional in the strict sense, but regardless it
    implements a `proximal` method and is hence usable with proximal solvers.
    See [Heide+2015] for more information.

    The functional requires an appropriate backend. To install the backends run

    ===========  ===============================
    `impl`       call
    ===========  ===============================
    `'skimage'`  ``$ pip install scikit-image``
    `'opencv'`   ``$ pip install opencv-python``
    ===========  ===============================

    Notes
    -----
    The nonlocal means regularization of a image :math:`u` is given by

    .. math::
        NL[u](x) =
            \frac{1}{C(x)}
            \int_\Omega
            e^{-\frac{(G_a * |u(x + \cdot) - u(y + \cdot)|^2)(0)}{h^2}}
            u(y) dy

    where :math:`\Omega` is the domain, :math:`G_a` is a gaussian kernel,
    :math:`h` is a parameter and :math:`*` denotes convolution and :math:`C(x)`
    is a normalization constant

    .. math::
        C(x) =
        \int_\Omega
        e^{-\frac{(G_a * |u(x + \cdot) - u(y + \cdot)|^2)(0)}{h^2}}
        dy

    See [Buades+2005] for more information.

    References
    ----------
    [Buades+2005] *A non-local algorithm for image denoising*, A. Buades,
    B. Coll and J.-M. Morel. CVPR 2005

    [Heide+2015] *FlexISP: a flexible camera image processing framework*,
    F. Heide et. al. SIGGRAPH Asia 2014
    """

    def __init__(self, space, h,
                 patch_size=7, patch_distance=11, impl='skimage'):
        """Initialize a new instance.
        """
        self.h = float(h)
        self.impl = impl
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        super(NLMRegularizer, self).__init__(
            space=space, linear=False, grad_lipschitz=np.nan)

    @property
    def proximal(self):
        func = self

        class NLMProximal(Operator):
            def __init__(self, stepsize):
                super(NLMProximal, self).__init__(
                    func.domain, func.domain, linear=False)
                self.stepsize = stepsize

            def _call(self, x):
                h = func.h * self.stepsize

                if func.impl == 'skimage':
                    from skimage.restoration import denoise_nl_means
                    x_arr = x.asarray()
                    return denoise_nl_means(
                        x_arr,
                        patch_size=func.patch_size,
                        patch_distance=func.patch_distance,
                        h=h,
                        multichannel=False)
                elif func.impl == 'opencv':
                    import cv2
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
    from odl.util.testutils import run_doctests
    run_doctests()
