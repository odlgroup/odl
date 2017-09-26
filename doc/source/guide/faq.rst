.. _FAQ:

##########################
Frequently asked questions
##########################

Abbreviations: **Q** uestion -- **P** roblem -- **S** olution

General errors
--------------

#. **Q:** When importing ``odl``, the following error is shown::

      File "/path/to/odl/odl/__init__.py", line 36

        from . import diagnostics

        ImportError: cannot import diagnostics

   However, I did not change anything in ``diagnostics``? Where does the error come from?

   **P:** Usually, this error originates from invalid code in a completely different place. You
   may have edited or added a module and broken the import chain in some way. Unfortunately, the
   error message is always as above, not specific to the invalid module.

   Another more subtle reason can be related to old
   `bytecode <https://en.wikipedia.org/wiki/Bytecode>`_ files. When you for the first time import
   (=execute) a module or execute a script, a bytecode file is created, basically to speed up
   execution next time. If you installed ``odl`` with ``pip -e`` (``--editable``), these files can
   sometimes interfere with changes to your codebase.

   **S:** Here are two things you can do to find the error more quickly.

   1. Delete the bytecode files. In a standard GNU/Linux shell, you can simply invoke (in your
      ``odl`` working directory)

      .. code-block:: bash

         find . -name *.pyc | xargs rm

   2. Execute the modules you changed since the last working (importable) state. In most IDEs, you
      have the possibility to run a currently opened file. Alternatively, you can run on the
      command line

      .. code-block:: bash

         python path/to/your/module.py

      This will yield a specific error message for an erroneous module that helps you debugging your
      changes.

#. **Q:** When adding two space elements, the following error is shown::

      TypeError: unsupported operand type(s) for +: 'DiscreteLpElement' and 'DiscreteLpElement'

   This seems completely illogical since it works in other situations and clearly must be supported.
   Why is this error shown?

   **P:** The elements you are trying to add are not in the same space.
   For example, the following code triggers the same error:

      >>> x = odl.uniform_discr(0, 1, 10).one()
      >>> y = odl.uniform_discr(0, 1, 11).one()
      >>> x - y

   In this case, the problem is that the elements have a different number of entries.
   Other possible issues include that they are discretizations of different sets,
   have different data types (:term:`dtype`), or implementation (for example CUDA/CPU).

   **S:** The elements need to somehow be cast to the same space.
   How to do this depends on the problem at hand.
   To find what the issue is, inspect the ``space`` properties of both elements.
   For the above example, we see that the issue lies in the number of discretization points:

      >>> x.space
      odl.uniform_discr(0, 1, 10)
      >>> y.space
      odl.uniform_discr(0, 1, 11)

   * In the case of spaces being discretizations of different underlying spaces,
     a transformation of some kind has to be applied (for example by using an operator).
     In general, errors like this indicates a conceptual issue with the code,
     for example a "we identify X with Y" step has been omitted.

   * If the ``dtype`` or ``impl`` do not match, they need to be cast to each one of the others.
     The most simple way to do this is by using the `DiscreteLpElement.astype` method.

#. **Q:** I have installed ODL with the ``pip install --editable`` option, but I still get an
   ``AttributeError`` when I try to use a function/class I just implemented. The use-without-reinstall
   thing does not seem to work. What am I doing wrong?

   **P:** You probably use an IDE like `Spyder`_ with integrated editor, console, etc. While your
   installation of the ODL *package* sees the changes immediately, the console still sees the
   version of the package *before the changes since it was opened*.

   **S:** Simply close the current console and open a new one.

Errors related to Python 2/3
----------------------------

#. **Q:** I follow your recommendation to call ``super().__init__(domain, range)`` in the ``__init__()`` method of ``MyOperator``, but I get the following error::

       File <...>, line ..., in __init__
		    super().__init__(dom, ran)

	   TypeError: super() takes at least 1 argument (0 given)

   What is this error related to and how can I fix it?

   **P:** The ``super()`` function `in Python 2 <https://docs.python.org/2/library/functions.html#super>`_ has to be called with a type as first argument, whereas `in Python 3    <https://docs.python.org/3/library/functions.html#super>`_, the type argument is optional and usually not needed.

   **S:** We recommend to use the explicit ``super(MyOperator, self)`` since it works in both Python 2 and 3.


Usage
-----

#. **Q:** I want to write an `Operator` with two input arguments, for example

   .. math::
      op(x, y) := x + y

   However, ODL only supports single arguments. How do I do this?

   **P:** Mathematically, such an operator is defined as

   .. math::
      \mathcal{A}: \mathcal{X}_1 \times \mathcal{X}_2
      \rightarrow \mathcal{Z}

   ODL adhers to the strict definition of this and hence only takes one parameter
   :math:`x \in \mathcal{X}_1 \times \mathcal{X}_2`. This product space element
   :math:`x` is then a tuple of elements :math:`x = (x_1, x_2),
   x_1 \in \mathcal{X}_1, x_2 \in \mathcal{X}_2`.

   **S:** Make the domain of the operator a `ProductSpace` if
   :math:`\mathcal{X}_1` and :math:`\mathcal{X}_2` are `LinearSpace`'s, or a
   `CartesianProduct` if they are mere `Set`'s. Mathematically, this
   corresponds to

   .. math::
      op([x, y]) := x + y

   Of course, a number of input arguments larger than 2 can be treated
   analogously.


.. _Spyder: https://github.com/spyder-ide/spyder
