# Copyright 2014, 2015 The ODL development group
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

from __future__ import print_function, division
from future import standard_library
standard_library.install_aliases()
from builtins import super

from skimage.transform import radon, iradon
import numpy as np
import odl

"""An example of a very SART tomography solver."""

class ForwardProjector(odl.Operator):
    def __init__(self, dom, ran):
        self.theta = ran.grid.meshgrid()[1][0] * 180 / np.pi
        super().__init__(dom, ran, True)

    def _call(self, x):
        return self.range.element(radon(x.asarray(), self.theta))

    @property
    def adjoint(self):
        return BackProjector(self.range, self.domain)

class BackProjector(odl.Operator):
    def __init__(self, dom, ran):
        self.theta = dom.grid.meshgrid()[1][0] * 180 / np.pi
        self.npoint = ran.grid.shape[0]
        super().__init__(dom, ran, True)

    def _call(self, x):
        return self.range.element(iradon(x.asarray(), self.theta, self.npoint, filter=None))

square = odl.Rectangle([-1, -1],[1, 1])
sinogram = odl.Rectangle([0, 0],[1, np.pi])

dom = odl.l2_uniform_discretization(odl.L2(square), [100, 100])
ran = odl.l2_uniform_discretization(odl.L2(sinogram), [142, 100])

volume = np.zeros([100, 100])
volume[40:-40,40:-40] = 1
x = dom.element(volume)

proj = ForwardProjector(dom, ran)

a = dom.one()

a+a
"""

data = proj(x)

x0 = dom.zero()
for i in range(50):
    x0 = x0 - proj.adjoint(proj(x0) - data) / 100
    x0.show()
"""