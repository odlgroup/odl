.. _numpy_in_depth:

##############################
Using ODL with NumPy and SciPy
##############################

`NumPy <http://www.numpy.org/>`_ is the traditional library for array computations in Python, and is still used by most major numerical packages.
It provides optimized `Array objects <http://docs.scipy.org/doc/numpy/reference/arrays.html>`_ that allow efficient storage of large arrays.
It also provides several optimized algorithms for many of the functions used in numerical programming, such as taking the cosine or adding two arrays.

`SciPy <http://www.scipy.org/>`_ is a library built on top of NumPy providing more advanced algorithms such as linear solvers, statistics, signal and image processing etc.

Many operations are more naturally performed using NumPy/SciPy than with ODL, and with that in mind ODL has been designed such that interfacing with them is as easy and fast as possible.

Casting vectors to and from arrays
==================================
ODL vectors are stored in an abstract way, enabling storage on the CPU, GPU, using different backends which can be switched using the `impl` argument when declaring the space.
This allows algorithms to be written in a generalized and storage-agnostic manner.
Still, it is often convenient to be able to access the raw data either for inspection or manipulation, perhaps to initialize a vector, or to call an external function.

To cast a NumPy array to an element of an ODL vector space, one can simply call the `LinearSpace.element` method in an appropriate space::

   >>> import odl
   >>> import numpy as np
   >>> r3 = odl.rn(3)
   >>> arr = np.array([1, 2, 3])
   >>> x = r3.element(arr)

Indeed, this works also for raw arrays of any library supporting the DLPack standard.
Note that this is not necessarily a good idea: for one thing, it will in general incur copying of data between different devices (which can take considerable time); for another, DLPack support is still somewhat inconsistent in libraries such as PyTorch as of 2025.

If the data type and storage methods allow it, copying is however avoided by default, and the element simply wraps the underlying array using a `view
<http://docs.scipy.org/doc/numpy/glossary.html#term-view>`_::

   >>> float_arr = np.array([1.0, 2.0, 3.0])
   >>> x = r3.element(float_arr)
   >>> x.data is float_arr
   True

..
  TODO the above is currently not satisfied (the array is copied, possibly due to a DLPack
  inconsistency). Fix?

Casting ODL vector space elements to NumPy arrays can be done through the member function `Tensor.asarray`. These returns a view if possible::

   >>> x.asarray()
   array([ 1.,  2.,  3.])

`Tensor.asarray` only yields a NumPy array if the space has `impl='numpy'` (the default).
If for example `impl='pytorch'`, it gives a `torch.Tensor` instead.

These methods work with any ODL object represented by an array.
For example, in discretizations, a two-dimensional array can be used::

   >>> space = odl.uniform_discr([0, 0], [1, 1], shape=(3, 3))
   >>> arr = np.array([[1, 2, 3],
   ...                 [4, 5, 6],
   ...                 [7, 8, 9]])
   >>> x = space.element(arr)
   >>> x.asarray()
   array([[ 1.,  2.,  3.],
          [ 4.,  5.,  6.],
          [ 7.,  8.,  9.]])

Using ODL objects with array-based functions
============================================
Although ODL offers its own interface to formulate mathematical algorithms (which we recommend using), there are situations where one needs to manipulate objects on the raw array level.

.. note::
  ODL versions 0.7 and 0.8 allowed directly applying NumPy ufuncs to ODL objects.
   This is not allowed anymore in ODL 1.x, since the ufunc compatibility mechanism interfered with high-performance support for other backends.

..
  TODO link to migration guide.

Apart from unwrapping the contained arrays and `.element`-wrapping their modified versions again (see above), there is also the option to modify as space element in-place using some NumPy function (or any function defined on backend-specific arrays). For this purpose we have the `writable_array` context manager that exposes a raw array which gets automatically assigned back to the ODL object::

    >>> x = odl.rn(3).element([1,2,3])
    >>> with odl.util.writable_array(x) as x_arr:
    ...     np.cumsum(x_arr, out=x_arr)
    >>> x
    rn(3).element([ 1.,  3.,  6.])

.. note::
    The re-assignment is a no-op if ``x`` has a single array as its data container, hence the operation will be as fast as manipulating ``x`` directly.
    The same syntax also works with other data containers, but in this case, copies are usually necessary.


NumPy functions as Operators
============================
It is often useful to write an `Operator` wrapping NumPy or other low-level functions, thus allowing full access to the ODL ecosystem.
The convolution operation, written as ODL operator, could look like this::

   >>> class MyConvolution(odl.Operator):
   ...     """Operator for convolving with a given kernel."""
   ...
   ...     def __init__(self, kernel):
   ...         """Initialize the convolution."""
   ...         self.kernel = kernel
   ...
   ...         # Initialize operator base class.
   ...         # This operator maps from the space of vector to the same space and is linear
   ...         super(MyConvolution, self).__init__(
   ...             domain=kernel.space, range=kernel.space, linear=True)
   ...
   ...     def _call(self, x):
   ...         # The output of an Operator is automatically cast to an ODL object
   ...         return self.range.element(np.convolve(x.asarray(), self.kernel.asarray(), mode='same'))

This operator can then be called on its domain elements::

   >>> kernel = odl.rn(3).element([1, 2, 1])
   >>> conv_op = MyConvolution(kernel)
   >>> conv_op([1, 2, 3])
   rn(3).element([ 4.,  8.,  8.])

N.B. the input list `[1,2,3]` is automatically wrapped into `conv_op.domain.element` by the `Operator` base class before the low-level call; in production code it is recommended to do this explicitly for better control.

Such operators can also be used with any of the ODL operator functionalities such as multiplication with scalar, composition, etc::

   >>> scaled_op = 2 * conv_op  # scale output by 2
   >>> scaled_op([1, 2, 3])
   rn(3).element([  8.,  16.,  16.])
   >>> y = odl.rn(3).element([1, 1, 1])
   >>> inner_product_op = odl.InnerProductOperator(y)
   >>> # Create composition with inner product operator with [1, 1, 1].
   >>> # When called on a vector, the result should be the sum of the
   >>> # convolved vector.
   >>> composed_op = inner_product_op @ conv_op
   >>> composed_op([1, 2, 3])
   20.0

For more information on ODL Operators, how to implement them and their features, see the guide on `operators_in_depth`.

Using ODL with SciPy linear solvers
===================================
SciPy includes `a series of very competent solvers <http://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_ that may be useful in solving some linear problems.
If you have invested some effort into writing an ODL operator, or perhaps wish to use a pre-existing operator, then the function `as_scipy_operator` creates a Python object that can be used in SciPy's linear solvers.
Here is a simple example of solving Poisson's equation :math:`- \Delta u = f` on the interval :math:`[0, 1]`::

   >>> space = odl.uniform_discr(0, 1, 5)
   >>> op = -odl.Laplacian(space)
   >>> f = space.element(lambda x: (x > 0.4) & (x < 0.6))  # indicator function on [0.4, 0.6]
   >>> u, status = scipy.sparse.linalg.cg(odl.as_scipy_operator(op), f.asarray())
   >>> u
   array([ 0.02,  0.04,  0.06,  0.04,  0.02])

Of course, this also could (and should!) be done with ODL's own version of the solver:

   >>> x = op.domain.element()
   >>> odl.solvers.conjugate_gradient(op=op, x=x, rhs=f, niter=100)
   >>> x
   uniform_discr(0.0, 1.0, 5).element([ 0.02,  0.04,  0.06,  0.04,  0.02])
