.. _glossary:

########
Glossary
########

.. glossary::

discretization
    Structure to handle the mapping from abstract objects (e.g. functions) to
    concrete realizations. It is a tuple with four elements:

    .. math::
        \mathcal{D}(\mathcal{X}) = (\mathcal{X}, \mathbb{F}^n,
        \mathcal{R}_\mathcal{X}, \mathcal{E}_\mathcal{X}),

    where :math:`\mathcal{X}` is the undiscretized set, :math:`\mathbb{F}^n`
    is a set of :math:`n`-tuples and

    .. math::
        \mathcal{R}_\mathcal{X}: \mathcal{X} \to \mathbb{F}^n,

        \mathcal{E}_\mathcal{X}: \mathbb{F}^n \to \mathcal{X}

    the :term:`restriction` and :term:`extension` mappings of this
    discretization of :math:`\mathcal{X}`.

    See :term:`restriction` and :term:`extension` for examples.

extension
    Operator in a :term:`discretization` mapping a concrete
    (finite-dimensional) object to an abstract (infinite-dimensional) one.

    **Example:**

    Let :math:`\mathcal{X} = C([0, 1])` be the space of real-valued
    continuous functions on the interval :math:`[0, 1]`. Let points
    :math:`x_1, \dots, x_n \in [0, 1]` and values
    :math:`\bar f = (f_1, \dots, f_n) \in \mathbb{R}^n` be given. Consider the
    linear interpolation at a value :math:`x \in [0, 1]`:

    .. math::
        I(\bar f; x) := (1 - \lambda(x)) f_i + \lambda(x) f_{i+1},

        \lambda(x) = \frac{x - x_i}{x_{i+1} - x_i},

    where :math:`i` is the index such that :math:`x \in [x_i, x_{i+1})`.

    Then the linear interpolation operator is defined as

    .. math::
        \mathcal{L} : \mathbb{R}^n \to C([0, 1]),

        \mathcal{L}(\bar f) := I(\bar f; \cdot),

    where :math:`I(\bar f; \cdot)` stands for the function
    :math:`x \mapsto I(\bar f; x)`.

    The abstract object in this case is the interpolatintg function
    :math:`I(\bar f; \cdot)`, created by the interpolation operator from
    the vector :math:`\bar f \in \mathbb{R}^n`.

restriction
    Operator in a :term:`discretization` mapping an abstract
    (infinite-dimensional) object to a concrete (finite-dimensional) one.

    **Example:**

    Let :math:`\mathcal{X} = C([0, 1])` be the space of real-valued
    continuous functions on the interval :math:`[0, 1]`. Let
    :math:`x_1, \dots, x_n` points in :math:`[0, 1]`. Then the
    *grid collocation* is defined by

    .. math::
        \mathcal{C}: \mathcal{X} \to \mathbb{R}^n,

        \mathcal{C}(f) := \big(f(x_1), \dots, f(x_n)\big).

    The abstract object in this case is the input function :math:`f`, and
    the operator evaluates this function at the given points, resulting in
    a vector in :math:`\mathbb{R}^n`.
    
vectorization
    Ability of a function to be evaluated on a grid in a single call rather
    than looping over the grid points. Vectorized evaluation gives a huge
    performance boost compared to Python loops (at least if there is no
    JIT) since loops are implemented in optimized C code.

    The vectorization concept in ODL differs slightly from the one in NumPy
    in that arguments have to be passed as a single tuple rather than a
    number of (positional) arguments. See :ref:`vectorization` for more
    details.
