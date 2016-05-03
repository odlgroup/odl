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

# Internal
import odl


class IntensityOperator(odl.discr.PointwiseTensorFieldOperator):

    """Intensity mapping of a complex wave function.

    A complex wave function is interpreted as a product space
    element ``x in X^2``, where ``X`` is a discretized function
    space. It maps a pair ``(f1, f2)`` to

        ``I(f1, f2) = |1 + (f1 + i*f2)|^2``

    where ``k`` is the wave number of the incoming plane wave and
    ``d`` the propagation distance.

    TODO: perhaps separate out the exponential part
    """

    def __init__(self, wavenum, propdist, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        wavenum : positive `float`
            Wave number (= 2*pi / (wave length)) of the incoming plane
            wave
        propdist : nonnegative `float`
            Distance f

        """


