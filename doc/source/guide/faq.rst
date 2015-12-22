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

  **P:** When you for the first time import (=execute) a module or execute a
  script, a `bytecode <https://en.wikipedia.org/wiki/Bytecode>`_ file is created,
  basically to speed up execution next time. If you installed ``odl`` with
  ``pip -e`` (``--editable``), these files can interfere with changes to your
  codebase.

  **S:** Delete the bytecode files. In a standard GNU/Linux shell, you can
  simply invoke (in your ``odl`` working directory)
  ::

    find . -name *.pyc | xargs rm


Errors related to Python 2/3
----------------------------

#. **Q:** I follow your recommendation to call ``super().__init__(dom, ran)``
   in the ``__init__()`` method of ``MyOperator``, but I get the following
   error::
   
	File <...>, line ..., in __init__
		super().__init__(dom, ran)

	TypeError: super() takes at least 1 argument (0 given)

   **P:** The ``super()`` function `in Python 2
   <https://docs.python.org/2/library/functions.html#super>`_ has to
   be called with a type as first argument, whereas
   `in Python 3
   <https://docs.python.org/3/library/functions.html#super>`_, the
   type argument is optional and usually not needed.

   **S:** We recommend to include ``from builtins import super`` in your
   module to backport the new Py3 ``super()`` function. This way, your code
   will run in both Python 2 and 3.
   
   
Usage
-----

#. **Q:** I want to write an `Operator` of with two input arguments, for example
   
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
