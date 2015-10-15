import odl

print('\n\n TESTING FOR L2 SPACE \n\n')

spc = odl.L2(odl.Interval(0,1))
disc = odl.l2_uniform_discretization(spc, 10)

odl.diagnostics.SpaceTest(disc).run_tests()

print('\n\n TESTING FOR Rn SPACE \n\n')

spc = odl.Rn(10)
odl.diagnostics.SpaceTest(spc).run_tests()


print('\n\n TESTING FOR Cn SPACE \n\n')

spc = odl.Cn(10)
odl.diagnostics.SpaceTest(spc).run_tests()


if odl.CUDA_AVAILABLE:
    print('\n\n TESTING FOR CudaRn SPACE \n\n')

    spc = odl.CudaRn(10)
    odl.diagnostics.SpaceTest(spc, eps=0.0001).run_tests()