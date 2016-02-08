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

"""An example of a very simple SART tomography solver.

Requires scikit-image (http://scikit-image.org/)
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super, range

# External
from skimage.transform import radon, iradon
import numpy as np

# Internal
import odl


class ForwardProjector(odl.Operator):
    def __init__(self, dom, ran):
        self.theta = ran.grid.meshgrid[1][0] * 180 / np.pi
        super().__init__(dom, ran, True)

    def _call(self, x):
        return self.range.element(radon(x.asarray(), self.theta))

    @property
    def adjoint(self):
        return BackProjector(self.range, self.domain)


class BackProjector(odl.Operator):
    def __init__(self, dom, ran):
        self.theta = dom.grid.meshgrid[1][0] * 180 / np.pi
        self.npoint = ran.grid.shape[0]
        super().__init__(dom, ran, True)

    def _call(self, x):
        return self.range.element(iradon(x.asarray(), self.theta, self.npoint,
                                         filter=None))


dom = odl.uniform_discr([-1, -1], [1, 1], [100, 100])
ran = odl.uniform_discr([0, 0], [1, np.pi], [142, 100])

proj_op = ForwardProjector(dom, ran)

phantom = odl.util.shepp_logan(dom, modified=True)
data = proj_op(phantom)

x = dom.zero()

for i in range(50):
    x = x - proj_op.adjoint(proj_op(x) - data) / 200
    print(i)

x.show()
