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

"""Visualization of the test functions in the diagnostics module."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import matplotlib.pyplot as plt

# Internal
import odl


disc = odl.uniform_discr(0, 1, 100)

for name, vec in odl.diagnostics.vector_examples(disc):
    vec.show(title=name)

disc = odl.uniform_discr([0, 0], [1, 1], [100, 100])

for name, vec in odl.diagnostics.vector_examples(disc):
    vec.show(title=name)

plt.show()
