"""An example of a very simple operator on rn."""

from __future__ import print_function
import odl


class AddOp(odl.Operator):
    def __init__(self, space, add_this):
        super(AddOp, self).__init__(domain=space, range=space)
        self.add_this = add_this

    def _call(self, x):
        return x + self.add_this


r3 = odl.rn(3)
x = r3.element([1, 2, 3])

op = AddOp(r3, add_this=10)

print(op(x))
