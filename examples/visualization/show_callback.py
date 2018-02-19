"""Example of using `odl.solvers.CallbackShow`."""

import odl

# These may show up very quickly, in order to get a good view of the images
# an `odl.solvers.CallbackSleep` can be used. Remove for faster plotting.
sleep = odl.solvers.CallbackSleep(seconds=1.0)

# Callback in 1d adds new lines to the figure
space_1d = odl.uniform_discr(0, 1, 100)
callback = sleep & odl.solvers.CallbackShow()
for name, elem in space_1d.examples:
    callback(elem)

# Callback in 2d replaces the figure in place
space_2d = odl.uniform_discr([0, 0], [1, 1], [100, 100])
callback = sleep & odl.solvers.CallbackShow()
for name, elem in space_2d.examples:
    callback(elem)
