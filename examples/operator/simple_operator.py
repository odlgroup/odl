"""An example of a very simple operator on rn."""

import odl


class AddOp(odl.Operator):
    def __init__(self, space, add_this):
        odl.Operator.__init__(self, domain=space, range=space)
        self.add_this = add_this

    def _call(self, x, out):
        return x + self.add_this


r3 = odl.rn(3)
x = r3.element([1, 2, 3])

op = AddOp(r3, add_this=10)

print(op(x))
