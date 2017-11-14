"""Visualization of the test functions in the diagnostics module."""

import odl
import matplotlib.pyplot as plt

space_1d = odl.uniform_discr(0, 1, 100)

for name, elem in space_1d.examples:
    elem.show(name)

space_2d = odl.uniform_discr([0, 0], [1, 1], [100, 100])

for name, elem in space_2d.examples:
    elem.show(name)

plt.show()
