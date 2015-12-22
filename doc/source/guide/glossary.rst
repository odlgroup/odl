.. _glossary:

########
Glossary
########

.. glossary::

array-like
    Any data structure which can be converted into a `numpy.ndarray` by the
    `numpy.array` constructor. Includes all `NtuplesBaseVector` based classes.

discretization
    Structure to handle the mapping between abstract objects (e.g. functions) and
    concrete, finite realization. It encompasses an abstract `Set`, a finite data
    container (`NtuplesBaseVector` in general) and the mappings between them,
    :term:`restriction` and :term:`extension`.

domain
    Set of elements to which an operator can be applied.

element-like
    Any data structure which can be converted into a ``<set>Vector`` by
    the ``<set>.element`` method.
    
    Example: ```DiscreteLp` element-like`` means that
    `DiscreteLp.element` can create a `DiscreteLpVector` from the input.

extension
    Operator in a :term:`discretization` mapping a concrete
    (finite-dimensional) object to an abstract (infinite-dimensional) one.
    Example: `LinearInterpolation`.

in-place evaluation
    Operator evaluation method which uses an existing data container to store
    the result. Usually more efficient than :term:`out-of-place evaluation`
    since no new memory is allocated and no data is copied.

out-of-place evaluation
    Operator evaluation method which creates a new data container to store
    the result. Usually less efficient than :term:`in-place evaluation`
    since new memory is allocated and data needs to be copied.

range
    Set of elements to which an operator maps, i.e. in which the result of
    an operator evaluation lies.

restriction
    Operator in a :term:`discretization` mapping an abstract
    (infinite-dimensional) object to a concrete (finite-dimensional) one.
    Example: `GridCollocation`.

vectorization
    Ability of a function to be evaluated on a grid in a single call rather
    than looping over the grid points. Vectorized evaluation gives a huge
    performance boost compared to Python loops (at least if there is no
    JIT) since loops are implemented in optimized C code.

    The vectorization concept in ODL differs slightly from the one in NumPy
    in that arguments have to be passed as a single tuple rather than a
    number of (positional) arguments. See :ref:`vectorization` for more
    details.
