"""Example on using `ProductSpaceElement.show`."""

import odl
import numpy as np

n = 100
m = 7
space = odl.uniform_discr([0, 0], [1, 1], [n, n])
pspace = odl.ProductSpace(space, m)

# Making a product space element where each component consists of a
# Shepp-Logan phantom multiplied by the constant i, where i is the
# index of the product space component.
elem = pspace.element([odl.phantom.shepp_logan(space, modified=True) * i
                       for i in range(m)])

# By default 4 uniformly spaced elements are shown. Since there are 7 in
# total, the shown components are 0, 2, 4 and 6
elem.show('Default')

# One can also use indexing by a list of indices or a slice.
elem.show('The first 2 elements', indices=[0, 1], force_show=True)

elem.show('Every third element', indices=np.s_[::3], force_show=True)

# Slices propagate (as in numpy): the first index in the slice applies to
# the product space components, the other dimensions are applied to each
# component. Here we take the second component and slice in the
# middle along the second axis.
elem.show('Element at index 2, sliced by [:, n // 2]',
          indices=[2, None, n // 2], force_show=True)
