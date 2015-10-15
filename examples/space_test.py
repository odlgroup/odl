import odl

print('\n\n TESTING FOR L2 SPACE \n\n')

spc = odl.L2(odl.Interval(0,1))
disc = odl.l2_uniform_discretization(spc, 10)

odl.diagnostics.SpaceTest(disc).run_tests()