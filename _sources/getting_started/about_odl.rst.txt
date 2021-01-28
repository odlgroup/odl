.. _about_odl:

#########
About ODL
#########

Operator Discretization Library (ODL) is a Python library for fast prototyping focusing on (but not restricted to) inverse problems.
ODL is being developed at `KTH Royal Institute of Technology, Stockholm`_, and `Centrum Wiskunde & Informatica (CWI), Amsterdam`_.

The main intent of ODL is to enable mathematicians and applied scientists to use different numerical methods on real-world problems without having to implement all necessary parts from the bottom up.
This is reached by an `Operator` structure which encapsulates all application-specific parts, and a high-level formulation of solvers which usually expect an operator, data and additional parameters.
The main advantages of this approach is that

1. Different problems can be solved with the same method (e.g. TV regularization) by simply switching operator and data.
2. The same problem can be solved with different methods by simply calling into different solvers.
3. Solvers and application-specific code need to be written only once, in one place, and can be tested individually.
4. Adding new applications or solution methods becomes a much easier task.


.. _KTH Royal Institute of Technology, Stockholm: https://www.kth.se/en/sci/institutioner/math
.. _Centrum Wiskunde & Informatica (CWI), Amsterdam: https://www.cwi.nl
