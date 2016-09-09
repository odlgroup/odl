"""Example of how to implement and use functionals."""

import odl


# Here we define the functional
class MyFunctional(odl.solvers.Functional):

    """This is my functional: ``||x||_2^2 + <x, y>``."""

    def __init__(self, space, y):
        """Initialize a new instance."""
        # This comand calls the init of Functional and sets a number of
        # parameters associated with a functional. All but domain have default
        # values if not set.
        odl.solvers.Functional.__init__(self, space=space, linear=False,
                                        grad_lipschitz=2)

        # We need to check that linear_term is in the domain. Then we store the
        # value of linear_term for future use.
        if y not in space:
            raise TypeError('linear_term is not in the domain!')
        self.y = y

    # Defining the _call function. This method is used for evaluation of
    # the functional and always needs to be implemented.
    def _call(self, x):
        """Evaluate the functional."""
        return x.norm()**2 + x.inner(self.y)

    # Next we define the gradient. Note that this is a property.
    @property
    def gradient(self):
        """The gradient operator."""

        # First we store the functional in a variable
        functional = self

        # The class corresponding to the gradient operator.
        class MyGradientOperator(odl.Operator):

            """Class implementing the gradient operator."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(domain=functional.domain,
                                 range=functional.domain)

            def _call(self, x):
                """Evaluate the gradient."""
                # Here we can access the store functional from a few lines
                # above
                return 2.0 * x + functional.y

        return MyGradientOperator()

    # Next we define the convex conjugate functional.
    @property
    def convex_conj(self):
        """The convex conjugate functional."""
        # This functional is implemented below.
        return MyFunctionalConjugate(space=self.domain, y=self.y)


# Here is the conjugate functional. Note that this is a separate class, in
# contrast to the gradient which was implemented as an inner class. One
# advantage with the inner class it that we don't have to pass as many
# parameters when initializing, on the other hand having separate classes
# normally improves readibility of the code. Both methods are use throughout
# the odl package.
class MyFunctionalConjugate(odl.solvers.Functional):

    """Conjugate functional to ``||x||_2^2 + <x,y>``.

    This funtional has the analytic expression

        ``f^*(x) = ||x-y||^2/4``.
    """

    def __init__(self, space, y):
        """initialize a new instance."""
        odl.solvers.Functional.__init__(self, space=space, linear=False,
                                        grad_lipschitz=2)

        if y not in space:
            raise TypeError('y is not in the domain!')
        self.y = y

    def _call(self, x):
        """Evaluate the functional."""
        return (x-self.y).norm()**2 / 4.0


# Create a functional
space = odl.uniform_discr(0, 1, 3)
linear_term = space.element([1, -4, 7])
my_func = MyFunctional(space=space, y=linear_term)

# Now we evaluate the functional in a random point
point = odl.util.testutils.noise_element(space)
print('Value of the functional in a random point: {}'
      ''.format(my_func(point)))

# Now we use the steepest-decent solver and backtracking linesearch in order to
# find the minimum of the functional.
# Create the gradient operator
grad = my_func.gradient

# Create a starting guess. Also used by the solver to update in-place.
x = space.one()

# Create the linesearch object
line_search = odl.solvers.BacktrackingLineSearch(my_func, max_num_iter=10)

# Call the solver
odl.solvers.steepest_descent(grad=grad, x=x, niter=10, line_search=line_search)

print('Expected value: {}'.format((-1.0 / 2) * linear_term))
print('Found value: {}'.format(x))

# Create the convex conjugate functional of a scaled and translated functional
scalar = 3.2
translation = space.one()
scal_trans_cc_func = (scalar * my_func).translated(translation).convex_conj

# Evaluating the new functional in the random point.
print('Value of the new functional in a random point: {}'
      ''.format(scal_trans_cc_func(point)))
