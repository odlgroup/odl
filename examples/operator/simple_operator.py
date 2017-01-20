"""An example of a very simple operator on rn."""

import odl


class AddOp(odl.Operator):
    def __init__(self, size, add_this):
        odl.Operator.__init__(self, domain=odl.rn(size), range=odl.rn(size))
        self.value = add_this

    def _call(self, x, out):
        out[:] = x.data + self.value

size = 3
rn = odl.rn(size)
x = rn.element([1, 2, 3])

op = AddOp(size, add_this=10)

print(op(x))
