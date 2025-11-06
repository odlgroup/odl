.. _migrate_0.x_to_1.x:

#################################
Migrating from ODL 0.x to ODL 1.x
#################################

If you have a project built around ODL versions 0.6, 0.7, 0.8, or built the
development version from the master branch until 2025 ("1.0-dev"), then you may
need to make some changes to use your code together with the official 1.0
release. This guide explains how.

NumPy ufuncs
============

The most significant change in 1.0 is in the way pointwise / elementwise functions
are applied to ODL objects (e.g. `DiscretizedSpaceElement`).
ODL 0.x ultimately stored all data of such an object inside one or multiple NumPy
arrays, and was thus able to hook into NumPy's "ufunc" mechanism to allow code like::

    >>> import odl # up to version 0.8.3
    >>> import numpy as np
    >>> space = odl.uniform_discr(0, np.pi, 7, nodes_on_bdry=True)
    >>> xs = space.element(lambda x: x)
    >>> np.cos(xs)
    uniform_discr(0.0, 3.1415927, 7, nodes_on_bdry=True).element(
        [ 1.       ,  0.8660254,  0.5      , ..., -0.5      , -0.8660254,
         -1.       ]
    )

If you run the same code with ODL 1.0, you will get an error message. The reason is
that ODL can now use other backends like PyTorch for storing the data, on which NumPy
ufuncs do not work. To offer a consistent way of performing pointwise operations on
such objects regardless of the backend, ODL now offers versions of these functions
in its own namespace:

    >>> # import odl from version 1.0
    >>> odl.cos(xs)
    uniform_discr(0.0, 3.1415927, 7, nodes_on_bdry=True).element(
        [ 1.       ,  0.8660254,  0.5      , ..., -0.5      , -0.8660254,
         -1.       ]
    )


Operator composition
====================

Operators are a central feature of ODL.
Typically, multiple primitive operators are composed to a whole pipeline.
ODL 0.x used Python's `*` for this purpose, which is intuitive from a
mathematical perspective particular for linear operators as composition
corresponds to matrix multiplication then.

Unfortunately it conflicted with another use of `*`, which most array libraries
employ, namely pointwise multiplication (for matrices, this is the Hadamard
product). To avoid mistakes from the different interpretations, from ODL 1.0 on
the `@` symbol should instead be used for composing operators (this is also used
by NumPy and PyTorch for matrix multiplication).
This also applies to the various ways ODL overloads "composition"; for example,
to pre-compose an :math:`L^2` norm with a pointwise scaling, you could write::

    >>> op = odl.functional.L2Norm(space) @ (1 + odl.sin(xs))
    >>> op
    FunctionalRightVectorMult(L2Norm(uniform_discr(0.0, 3.1415927, 7, nodes_on_bdry=True)), uniform_discr(0.0, 3.1415927, 7, nodes_on_bdry=True).element(
        [ 1.       ,  1.5      ,  1.8660254, ...,  1.8660254,  1.5      ,
          1.       ]
    ))
    >>> op(space.one())
    2.9360830109198384

In some cases, the old `*` syntax is still interpreted as composition when that
is unambiguous, but this is deprecated and should be replaced with `@`.
Only use `*` for multiplying odl objects pointwise, for example::

    >>> odl.sqrt(xs) * odl.sqrt(xs) - xs
    uniform_discr(0.0, 3.1415927, 7, nodes_on_bdry=True).element(
        [ 0.,  0.,  0., ...,  0.,  0., -0.]
    )
