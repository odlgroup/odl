.. _dev_extend:

#############
Extending ODL
#############

ODL is written to be easy to extend with new functionality and classes, and new content is welcome.
With that said, not everything fits inside the main library, and some ideas are better realized as *extension packages*, i.e., packages that use the core ODL library and extend it with experimental features.
This lowers the requirement on code maturity, completeness of documentation, unit tests etc. on your side and allows the core library to stay slim and develop faster.

There are several ways to extend ODL, some of which are listed below.

Adding Tensor spaces
--------------------
The abstract `TensorSpace` is the workhorse of the ODL space machinery. It is used in the discrete :math:`R^n` case, as well as data representation for discretized function spaces such as :math:`L^2([0, 1])` in the `DiscretizedSpace` class.
These are in general created through the `rn` and `uniform_discr` functions who take an ``impl`` parameter, allowing users to select the backend to use.

In the core ODL package, there is only a single backend available: `NumpyTensorSpace`, given by ``impl='numpy'``, which is the default choice.
Users can add CUDA support by installing the add-on library `odlcuda`_, which contains the additional space ``CudaFn``.
By using the `rn`/`uniform_discr` functions, users can then seamlessly change the backend of their spaces.

As an advanced user, you may need to add additional spaces of this type that can be used inside ODL, perhaps to add `MPI`_ support.
There are a few steps to do this:

* Create a new library with a ``setuptools`` installer in the form of a ``setup.py`` file.
* Add the spaces that you want to add to the library.
  The space needs to inherit from `TensorSpace` and implement all of the abstract methods in those spaces.
  See the spaces for further information on the specific methods that need to be implemented.
* Add the methods ``tensor_space_impls()`` to a file ``odl_plugin.py`` in your library.
  These should return a ``dict`` mapping implementation names to class names.
* Add the following to your library's ``setup.py`` setup call: ``entry_points={'odl.space': ['mylib = mylib.odl_plugin']``, where you replace ``mylib`` with the name of your plugin.

For a blueprint of all these steps, check out the implementation of the `odlcuda`_ plugin.

.. _odlcuda: https://github.com/odlgroup/odlcuda
.. _MPI: https://en.wikipedia.org/wiki/Message_Passing_Interface
