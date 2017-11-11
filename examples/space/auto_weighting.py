"""Example demonstrating the usage of the ``auto_weighting`` decorator."""

import odl
from odl.space.space_utils import auto_weighting


# --- Operator using auto-weighting --- #


class ScalingOp(odl.Operator):

    """Operator that scales input by a constant."""

    def __init__(self, dom, ran, c):
        super(ScalingOp, self).__init__(dom, ran, linear=True)
        self.c = c

    def _call(self, x):
        return self.c * x

    @property
    @auto_weighting
    def adjoint(self):
        return ScalingOp(self.range, self.domain, self.c)


# --- Test on Rn space --- #


print('')
print('*** Test with Rn type spaces ***')
print('')

rn = odl.rn(2)  # Constant weight 1
discr = odl.uniform_discr(0, 4, 2)  # Constant weight 2
print('*** Spaces ***')
print('X =', rn, '- weight =', rn.weighting.const)
print('Y =', discr, '- weight =', discr.weighting.const)
print('')

op1 = ScalingOp(rn, rn, 2)  # Same weightings, no scaling in adjoint
op2 = ScalingOp(discr, discr, 2)  # Same weightings, no scaling in adjoint
op3 = ScalingOp(rn, discr, 2)  # Different weightings, adjoint scales

# Look at output of ajoint
print('*** Ajoint evaluation ***')
print('X -> X adjoint at one:', op1.adjoint(op1.range.one()))
print('Y -> Y adjoint at one:', op2.adjoint(op2.range.one()))
print('X -> Y adjoint at one:', op3.adjoint(op3.range.one()))
print('')

# Check adjointness
print('*** Check adjointness ***')
inner1_dom = op1.domain.one().inner(op1.adjoint(op1.range.one()))
inner1_ran = op1(op1.domain.one()).inner(op1.range.one())
print('X -> X:    <Sx, y> = {},  <x, S^*y> = {}'
      ''.format(inner1_ran, inner1_dom))

inner2_dom = op2.domain.one().inner(op2.adjoint(op2.range.one()))
inner2_ran = op2(op2.domain.one()).inner(op2.range.one())
print('Y -> Y:    <Sx, y> = {},  <x, S^*y> = {}'
      ''.format(inner2_ran, inner2_dom))

inner3_dom = op3.domain.one().inner(op3.adjoint(op3.range.one()))
inner3_ran = op3(op3.domain.one()).inner(op3.range.one())
print('X -> Y:    <Sx, y> = {},  <x, S^*y> = {}'
      ''.format(inner3_ran, inner3_dom))


# --- Test on product spaces and array weighting --- #


print('')
print('')
print('*** Test with Product spaces ***')
print('')

pspace_1 = odl.ProductSpace(odl.rn(2), 3)  # Constant weight 1
pspace_2 = odl.ProductSpace(odl.rn(2), 3, weighting=[2, 1, 2])
print('*** Spaces ***')
print('X =', pspace_1, '- weight =', pspace_1.weighting.const)
print('Y =', pspace_2, '- weight =', repr(pspace_2.weighting.array))
print('')

op1 = ScalingOp(pspace_1, pspace_1, 2)  # Same weightings, no adjoint scaling
op2 = ScalingOp(pspace_2, pspace_2, 2)  # Same weightings, no adjoint scaling
op3 = ScalingOp(pspace_1, pspace_2, 2)  # Different weightings, adjoint scales

# Look at output of ajoint
print('*** Ajoint evaluation ***')
print('X -> X adjoint at one:', op1.adjoint(op1.range.one()))
print('Y -> Y adjoint at one:', op2.adjoint(op2.range.one()))
print('X -> Y adjoint at one:', op3.adjoint(op3.range.one()))
print('')

# Check adjointness
print('*** Check adjointness ***')
inner1_dom = op1.domain.one().inner(op1.adjoint(op1.range.one()))
inner1_ran = op1(op1.domain.one()).inner(op1.range.one())
print('X -> X:    <Sx, y> = {},  <x, S^*y> = {}'
      ''.format(inner1_ran, inner1_dom))

inner2_dom = op2.domain.one().inner(op2.adjoint(op2.range.one()))
inner2_ran = op2(op2.domain.one()).inner(op2.range.one())
print('Y -> Y:    <Sx, y> = {},  <x, S^*y> = {}'
      ''.format(inner2_ran, inner2_dom))

inner3_dom = op3.domain.one().inner(op3.adjoint(op3.range.one()))
inner3_ran = op3(op3.domain.one()).inner(op3.range.one())
print('X -> Y:    <Sx, y> = {},  <x, S^*y> = {}'
      ''.format(inner3_ran, inner3_dom))
