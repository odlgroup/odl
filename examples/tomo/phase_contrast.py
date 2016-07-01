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

"""Phase contrast reconstruction example."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.discr.lp_discr import DiscreteLp
from odl.discr.tensor_ops import PointwiseTensorFieldOperator
from odl.operator.default_ops import MultiplyOperator
from odl.operator.pspace_ops import ReductionOperator
from odl.space.pspace import ProductSpace


class IntensityOperator(PointwiseTensorFieldOperator):

    """Intensity mapping of a vectorial function."""

    def __init__(self, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : power space of `DiscreteLp`, optional
            The space of elements which the operator acts on. If
            ``range`` is given, ``domain`` must be a power space
            of ``range``.
        range : `DiscreteLp`, optional
            The space of elements to which the operator maps.
            This is required if ``domain`` is not given.

        Notes
        -----
        This operator maps a real vector field :math:`f = (f_1, \dots, f_d)`
        to its pointwise intensity

            :math:`\mathcal{I}(f) = \\lvert f\\rvert^2 :
            x \mapsto \sum_{j=1}^d f_i(x)^2`.

        """
        if domain is None and range is None:
            raise ValueError('either domain or range must be specified.')

        if domain is None:
            if not isinstance(range, DiscreteLp):
                raise TypeError('range {!r} is not a DiscreteLp instance.'
                                ''.format(range))
            domain = ProductSpace(range, 2)

        if range is None:
            if not isinstance(domain, ProductSpace):
                raise TypeError('domain {!r} is not a `ProductSpace` '
                                'instance.'.format(domain))
            range = domain[0]

        super().__init__(domain, range, linear=False)

    def _call(self, x, out):
        """Implement ``self(x, out)``."""
        out[:] = x[0]
        out *= out

        tmp = self.base_space.element()
        for xi in x[1:]:
            tmp.assign(xi)
            tmp *= tmp
            out += tmp

    def derivative(self, f):
        """Return the derivative operator in ``f``.

        Parameters
        ----------
        f : domain element
            Point at which the derivative is taken

        Returns
        -------
        deriv : `Operator`
            Derivative operator at the specified point

        Notes
        -----
        The derivative of the intensity operator is given by

            :math:`\partial \mathcal{I}(f_1, f_2)(h_1, h_2) =
            2 (f_1 h_1 + f_2 h_2)`.

        Its adjoint maps a function :math:`g` to the product space
        element

            :math:`\\left[\partial\mathcal{I}(f_1, f_2)\\right]^*(g) =
            2 (f_1 g, f_2 g)`.
        """
        mul_ops = [2 * MultiplyOperator(fi, domain=self.base_space)
                   for fi in f]
        return ReductionOperator(*mul_ops)


def propagation_kernel_ft_cos(x, **kwargs):
    """Modified Fresnel propagation kernel for the real part.

    Notes
    -----
    The kernel is defined as

        :math:`k(\\xi) = -\\frac{\kappa}{2}
        \cos\\left(\kappa d + \\frac{d}{2\kappa} \\lvert \\xi \\rvert^2
        \\right)`,

    where :math:`\kappa` is the wave number of the incoming wave and
    :math:`d` the propagation distance.
    """
    wavenum = float(kwargs.pop('wavenum', 1.0))
    prop_dist = float(kwargs.pop('prop_dist', 1.0))
    scaled = [np.sqrt(prop_dist / (2 * wavenum)) * xi for xi in x]
    kernel = sum(sxi ** 2 for sxi in scaled)
    kernel += wavenum * prop_dist
    np.cos(kernel, out=kernel)
    kernel *= -wavenum / 2
    return kernel


def propagation_kernel_ft_sin(x, **kwargs):
    """Modified Fresnel propagation kernel for the imaginary part.

    Notes
    -----
    The kernel is defined as

        :math:`k(\\xi) = -\\frac{\kappa}{2}
        \sin\\left(\kappa d + \\frac{d}{2\kappa} \\lvert \\xi \\rvert^2
        \\right)`,

    where :math:`\kappa` is the wave number of the incoming wave and
    :math:`d` the propagation distance.
    """
    wavenum = float(kwargs.pop('wavenum', 1.0))
    prop_dist = float(kwargs.pop('prop_dist', 1.0))
    scaled = [np.sqrt(prop_dist / (2 * wavenum)) * xi for xi in x]
    kernel = sum(sxi ** 2 for sxi in scaled)
    kernel += wavenum * prop_dist
    np.sin(kernel, out=kernel)
    kernel *= -wavenum / 2
    return kernel


#%% Example 1: Real and imaginary part

import odl

# Discrete reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_corner=[-20, -20, -20], max_corner=[20, 20, 20],
    nsamples=[300, 300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = (558, 558), min = (-30, -30), max = (30, 30)
detector_partition = odl.uniform_partition([-30, -30], [30, 30], [558, 558])
# Discrete reconstruction space

# Astra cannot handle axis aligned origin_to_det unless it is aligned
# with the third coordinate axis. See issue #18 at ASTRA's github.
# This is fixed in new versions of astra, with older versions, this could
# give a zero result.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

# ray transform aka forward projection. We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
vec_ray_trafo = odl.ProductSpaceOperator([[ray_trafo, 0],
                                          [0, ray_trafo]])

ft = odl.trafos.FourierTransform(ray_trafo.range, axes=[1, 2])
prop_sin = ft.range.element(propagation_kernel_ft_sin, wavenum=10000,
                            prop_dist=0.1)
prop_cos = ft.range.element(propagation_kernel_ft_cos, wavenum=10000,
                            prop_dist=0.1)

prop_1 = ft.inverse * prop_sin * ft
prop_2 = ft.inverse * prop_cos * ft

