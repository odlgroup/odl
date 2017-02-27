.. _numpy_in_depth:

##############################
Using ODL with NumPy and SciPy
##############################

`Numpy <http://www.numpy.org/>`_ is the ubiquitous library for array computations in Python, and is used by almost all major numerical packages.
It provides optimized `Array objects <http://docs.scipy.org/doc/numpy/reference/arrays.html>`_ that allow efficient storage of large arrays.
It also provides several optimized algorithms for many of the functions used in numerical programming, such as taking the cosine or adding two arrays.

`SciPy <http://www.scipy.org/>`_ is a library built on top of NumPy providing more advanced algorithms such as linear solvers, statistics, signal and image processing etc.

Many operations are more naturally performed using NumPy/SciPy than with ODL, and with that in mind ODL has been designed such that interfacing with them is as easy and fast as possible.

Casting vectors to and from arrays
==================================
ODL vectors are stored in an abstract way, enabling storage on the CPU, GPU, or perhaps on a cluster on the other side of the world.
This allows algorithms to be written in a generalized and storage-agnostic manner.
Still, it is often convenient to be able to access the data and look at it, perhaps to initialize a vector, or to call an external function.

To cast a NumPy array to an element of an ODL vector space, you simply need to call the `LinearSpace.element` method in an appropriate space::

   >>> r3 = odl.rn(3)
   >>> arr = np.array([1, 2, 3])
   >>> x = r3.element(arr)

If the data type and storage methods allow it, the element simply wraps the underlying array using a `view
<http://docs.scipy.org/doc/numpy/glossary.html#term-view>`_::

   >>> float_arr = np.array([1.0, 2.0, 3.0])
   >>> x = r3.element(float_arr)
   >>> x.data is float_arr
   True

Casting ODL vector space elements to NumPy arrays can be done in two ways, either through the member function `Tensor.asarray`, or using `numpy.asarray`. These are both optimized and if possible return a view::

   >>> x.asarray()
   array([ 1.,  2.,  3.])
   >>> np.asarray(x)
   array([ 1.,  2.,  3.])

These methods work with any ODL object represented by an array. For example, in discretizations, a two-dimensional array can be used::

   >>> space = odl.uniform_discr([0, 0], [1, 1], [3, 3])
   >>> arr = np.array([[1, 2, 3],
   ...                 [4, 5, 6],
   ...                 [7, 8, 9]])
   >>> x = space.element(arr)
   >>> x.asarray()
   array([[ 1.,  2.,  3.],
          [ 4.,  5.,  6.],
          [ 7.,  8.,  9.]])

Using ODL vectors with NumPy functions
======================================
A very convenient feature of ODL is its seamless interaction with NumPy functions. For universal functions (`ufuncs
<http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_) this is supported both via method of the ODL object and by direct application of the NumPy functions. For example, using NumPy::

   >>> r3 = odl.rn(3)
   >>> x = r3.element([1, 2, 3])
   >>> np.negative(x)
   rn(3).element([-1.0, -2.0, -3.0])

This method always uses the NumPy implementation, which can involve overhead in case the data is not stored in a CPU space. To always enable optimized code, users can call the member `Tensor.ufuncs`::

   >>> x.ufuncs.negative()
   rn(3).element([-1.0, -2.0, -3.0])

For other arbitrary functions, ODL vector space elements are generally accepted as input, but the output is often of `numpy.ndarray` type::

   >>> np.convolve(x, x, mode='same')
   array([  4.,  10.,  12.])

A known limitation of this approach is that ODL objects are not accepted as ``out`` parameters of Numpy functions.
For example, ``np.negative(x, out=y)`` will fail for two ODL space elements ``x, y``.
This is an issue with Numpy, and a `fix is underway <http://docs.scipy.org/doc/numpy-dev/neps/ufunc-overrides.html>`_, see also `this Numpy pull request <https://github.com/numpy/numpy/pull/8247>`_.
In the meantime, we work around this limitation in two ways.
First, we use the member ufuncs which accept an ``out`` parameter in an appropriate space::

    >>> y = x.space.element()  # uninitialized
    >>> result = x.ufuncs.negative(out=y)
    >>> result is y
    True
    >>> y
    rn(3).element([-1.0, -2.0, -3.0])

Another advantage of these member ufuncs is that they are ususally optimized for a specific implementation, for example CUDA-based arrays, while ``np.*`` always converts to a Numpy array first.
(This could potentially be changed with the planned ``__numpy_ufunc__`` interface, but currently it is not possible.)

Second, if a space element has to be modified in-place using some Numpy function (or any function defined on arrays), we have the `writable_array` context manager that exposes a Numpy array which gets automatically assigned back to the ODL object::

    >>> with odl.util.writable_array(x) as x_arr:
    ...     np.negative(x_arr, out=x_arr)
    >>> x
    rn(3).element([-1.0, -2.0, -3.0])

.. note::
    The re-assignment is a no-op if ``x`` has a Numpy array as its data container, hence the operation will be as fast as manipulating ``x`` directly.
    The same syntax also works with other data containers, but in this case, copies to and from a Numpy array are usually necessary.


NumPy functions as Operators
============================
To solve the above issue, it is often useful to write an `Operator` wrapping NumPy functions, thus allowing full access to the ODL ecosystem.
To wrap the convolution operation, you could write a new class::

   >>> class MyConvolution(odl.Operator):
   ...     """Operator for convolving with a given vector."""
   ...
   ...     def __init__(self, vector):
   ...         """Initialize the convolution."""
   ...         self.vector = vector
   ...
   ...         # Initialize operator base class.
   ...         # This operator maps from the space of vector to the same space and is linear
   ...         odl.Operator.__init__(self, domain=vector.space, range=vector.space,
   ...                               linear=True)
   ...
   ...     def _call(self, x):
   ...         # The output of an Operator is automatically cast to an ODL vector
   ...         return np.convolve(x, self.vector, mode='same')

This could then be called as an ODL Operator::

   >>> op = MyConvolution(x)
   >>> op(x)
   rn(3).element([4.0, 10.0, 12.0])

Since this is an ODL Operator, it can be used with any of the ODL functionalities such as multiplication with scalar, composition, etc::

   >>> scaled_op = 2 * op  # scale by scalar
   >>> scaled_op(x)
   rn(3).element([8.0, 20.0, 24.0])
   >>> y = r3.element([1, 1, 1])
   >>> inner_product_op = odl.InnerProductOperator(y)
   >>> composed_op = inner_product_op * op  # create composition with inner product with vector [1, 1, 1]
   >>> composed_op(x)
   26.0

For more information on ODL Operators, how to implement them and their features, see the guide on `operators_in_depth`.

Using ODL with SciPy linear solvers
===================================
SciPy includes `a series of very competent solvers <http://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_ that may be useful in solving some linear problems.
If you have invested some effort into writing an ODL operator, or perhaps wish to use a pre-existing operator then the function `as_scipy_operator` creates a Python object that can be used in SciPy's linear solvers.
Here is a simple example of solving Poisson's equation equation on an interval (:math:`- \Delta x = \text{rhs}`)::

   >>> space = odl.uniform_discr(0, 1, 5)
   >>> op = -odl.Laplacian(space)
   >>> rhs = space.element(lambda x: (x > 0.4) & (x < 0.6))  # indicator function on [0.4, 0.6]
   >>> result, status = scipy.sparse.linalg.cg(odl.as_scipy_operator(op), rhs)
   >>> result
   array([ 0.02,  0.04,  0.06,  0.04,  0.02])
