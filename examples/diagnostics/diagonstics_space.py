"""Run the standardized diagonstics suite on some of the spaces."""

import odl

print('\n\n TESTING FOR Lp SPACE \n\n')

discr = odl.uniform_discr(0, 1, 10)
odl.diagnostics.SpaceTest(discr).run_tests()

print('\n\n TESTING FOR rn SPACE \n\n')

spc = odl.rn(10)
odl.diagnostics.SpaceTest(spc).run_tests()


print('\n\n TESTING FOR cn SPACE \n\n')

spc = odl.cn(10)
odl.diagnostics.SpaceTest(spc).run_tests()


if 'cuda' in odl.space.entry_points.TENSOR_SPACE_IMPLS:
    print('\n\n TESTING FOR CUDA rn SPACE \n\n')

    spc = odl.rn(10, impl='cuda')
    odl.diagnostics.SpaceTest(spc, tol=0.0001).run_tests()
