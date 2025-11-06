.. _vectorization_in_depth:

####################
Vectorized functions
####################


This section is intended as a small guideline on how to write functions which work with the
vectorization machinery by low-level libraries which are used internally in ODL.


What is vectorization?
======================

In general, :term:`vectorization` means that a function can be evaluated on a whole array of values
at once instead of looping over individual entries. This is very important for performance in an
interpreted language like Python, since loops are usually very slow compared to compiled languages.


How to use NumPy's ufuncs
=========================

Until recently, the most common means of vectorization were the *uniform functions* from the `NumPy <http://www.numpy.org/>`_ library::

    def gaussian(x: np.ndarray):
        # Negation, powers and scaling are vectorized, of course.
        return np.exp(-x**2 / 2)

    def step(x: np.ndarray):
        # np.where checks the condition in the first argument and
        # returns the second for `True`, otherwise the third. The
        # last two arguments can be arrays, too.
        # Note that also the comparison operation is vectorized.
        return np.where(x[0] <= 0, 0, 1)

This covers a very large range of useful functions already (basic arithmetic is vectorized,
too!). Unfortunately, it is not compatible with GPU-based storage.

Other libraries offer a similar set of functions too, restricted to inputs from the same::

    def gaussian_torch(x: torch.Tensor):
        return torch.exp(-x**2 / 2)

The `Python Array API <https://data-apis.org/array-api>`_ is an attempt at unifying these functionalities, but it still requires selecting a *namespace* corresponding to a particular API-instantiation at the start::

    def gaussian_arr_api(x):
        xp = x.__array_namespace__()
        return xp.exp(-x**2 / 2)

Usage of raw-array functions in ODL
===================================

One use pointwise functions is as input to a discretization process. For example, we may
want to discretize a two-dimensional Gaussian function::

    >>> def gaussian2(x):
    ...     xp = x[0].__array_namespace__()
    ...     return xp.exp(-(x[0]**2 + x[1]**2) / 2)

on the rectangle [-5, 5] x [-5, 5] with 100 pixels in each
dimension. One way to do this is to pass the existing (raw-array based,
discretization-oblivious) function to the `DiscretizedSpace.element` method::

    >>> # Note that the minimum and maxiumum coordinates are given as
    >>> # vectors, not one interval at a time.
    >>> discr = odl.uniform_discr([-5, -5], [5, 5], (100, 100))

    >>> # This creates an element in the discretized space ``discr``
    >>> gaussian_discr = discr.element(gaussian2)

What happens behind the scenes is that ``discr`` creates a :term:`discretization` object which
has a built-in method ``element`` to turn continuous functions into discrete arrays by evaluating
them at a set of grid points. In the example above, this grid is a uniform sampling of the rectangle
by 100 points per dimension. ::

    >>> gaussian_discr.shape
    (100, 100)

To make this process fast, ``element`` assumes that the function is written in a way that not only
supports vectorization, but also guarantees that the output has the correct shape. The function
receives a :term:`meshgrid` tuple as input, in the above case consisting of two vectors::

    >>> mesh = discr.meshgrid
    >>> mesh[0].shape
    (100, 1)
    >>> mesh[1].shape
    (1, 100)

When inserted into the function, the final shape of the output is determined by Numpy's
`broadcasting rules`_ (or generally the Array API). For the Gaussian function, Numpy will conclude that the output shape must
be ``(100, 100)`` since the arrays in ``mesh`` are added after squaring. This size is the same
as expected by the discretization.

Pointwise functions on ODL objects
==================================

A perhaps more elegant alternative to the above is to start by generating ODL objects
corresponding only to primitive quantities, and then carry out the interesting computations
on those objects. This offers more type safety, and avoids the need to worry about any
array-namespaces::

    >>> r_sq = discr.element(lambda x: x[0]**2 + x[1]**2)
    >>> gaussian_discr = odl.exp(-r_sq/2)

In this case, `odl.exp` automatically resolves whichever array backend is
needed, as governed by the space::

    >>> discr = odl.uniform_discr([-5, -5], [5, 5], (100, 100), impl='pytorch')
    >>> r_sq = discr.element(lambda x: x[0]**2 + x[1]**2)
    >>> type(odl.exp(-r_sq/2).data)
    <class 'torch.Tensor'>

Further reading
===============

`Scipy Lecture notes on Numpy <http://www.scipy-lectures.org/intro/numpy/index.html>`_


.. _Universal functions (ufunc): http://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _available ufuncs: http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs
.. _special functions: http://docs.scipy.org/doc/scipy/reference/special.html
.. _broadcasting rules: http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
