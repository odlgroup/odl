.. _vectorization_in_depth:

####################
Vectorized functions
####################


This section is intended as a small guideline on how to write functions which work with the
vectorization machinery by Numpy which is used internally in ODL. 


What is vectorization?
======================

In general, :term:`vectorization` means that a function can be evaluated on a whole array of values
at once instead of looping over individual entries. This is very important for performance in an
interpreted language like python, since loops are usually very slow compared to compiled languages.

Technically, vectorization in Numpy works through the `Universal functions (ufunc)`_ interface. It
is fast because all loops over data are implemented in C, and the resulting implementations are
exposed to Python for each function individually.


How to use Numpy's ufuncs?
==========================

The easiest way to write fast custom mathematical functions in Python is to use the
`available ufuncs`_ and compose them to a new function::

    def gaussian(x):
        # Negation, powers and scaling are vectorized, of course.
        return np.exp(-x ** 2 / 2)

    def step(x):
        # np.where checks the condition in the first argument and
        # returns the second for `True`, otherwise the third. The
        # last two arguments can be arrays, too.
        # Note that also the comparison operation is vectorized.
        return np.where(x[0] <= 0, 0, 1)

This should cover a very large range of useful functions already (basic arithmetic is vectorized,
too!). An even larger list of `special functions`_ are available in the Scipy package.


Usage in ODL
============

Python functions are in most cases used as input to a discretization process. For example, we may
want to discretize a two-dimensional Gaussian function
::
    def gaussian2(x):
        return np.exp(-(x[0] ** 2 + x[1] ** 2) / 2)
    
on the rectangle [-5, 5] x [-5, 5] with 100 pixels in each
dimension. The code for this is simply
::
    # Note that the minimum and maxiumum coordinates are given as
    # vectors, not one interval at a time.
    discr = odl.uniform_discr([-5, -5], [5, 5], (100, 100))

    # This creates an element in the discretized space ``discr``
    gaussian_discr = discr.element(gaussian2)

What happens behind the scenes is that ``discr`` creates a :term:`discretization` object which
has a built-in method ``element`` to turn continuous functions into discrete arrays by evaluating
them at a set of grid points. In the example above, this grid is a uniform sampling of the rectangle
by 100 points per dimension.

To make this process fast, ``element`` assumes that the function is written in a way that not only
supports vectorization, but also guarantees that the output has the correct shape. The function
receives a :term:`meshgrid` tuple as input, in the above case consisting of two vectors::

    >>> mesh = discr.meshgrid()
    >>> mesh[0].shape
    (100, 1)
    >>> mesh[1].shape
    (1, 100)

When inserted into the function, the final shape of the output is determined by Numpy's
`broadcasting rules`_. For the Gaussian function, Numpy will conclude that the output shape must
be ``(100, 100)`` since the arrays in ``mesh`` are added after squaring. This size is the same
as expected by the discretization.

If, however, the function does not use all components of its input, the shape will be different::

    >>> def gaussian_const_x0_bad(x):
    ...     return np.exp(-x[1] ** 2 / 2)  # no x[0] -> no broadcasting

    >>> gaussian_const_x0_bad(mesh).shape
    (1, 100)

This array is too small for the discretization, and an exception will be raised, stating that this
function cannot be discretized.

The solution to this issue is rather simple: just make sure that all components are used such that
the broadcasting rules are triggered::

    >>> def gaussian_const_x0_good(x):
    ...     return np.exp(-x[1] ** 2 / 2) + 0 * x[0]  # broadcasting

    >>> gaussian_const_x0_good(mesh).shape
    (100, 100)



Further reading
===============

`Scipy Lecture notes on Numpy <http://www.scipy-lectures.org/intro/numpy/index.html>`_


.. _Universal functions (ufunc): http://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _available ufuncs: http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs
.. _special functions: http://docs.scipy.org/doc/scipy/reference/special.html
.. _broadcasting rules: http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
