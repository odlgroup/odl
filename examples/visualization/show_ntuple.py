"""Example for using `Tensor.show`.

The ``show`` method is implemented for all types of n-tuples and displays
the data as a scatter plot.
"""

import odl

space = odl.rn(5)
vector = space.element([1, 2, 3, 4, 5])
vector.show(force_show=True)
