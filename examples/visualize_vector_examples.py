import odl
import matplotlib.pyplot as plt

spc = odl.L2(odl.Interval(0,1))
disc = odl.l2_uniform_discretization(spc, 100)

for name, vec in odl.test.vector_examples(disc):
    vec.show(title=name)
    
spc = odl.L2(odl.Rectangle([0,0],[1,1]))
disc = odl.l2_uniform_discretization(spc, [100, 100])

for name, vec in odl.test.vector_examples(disc):
    vec.show(title=name)

plt.show()