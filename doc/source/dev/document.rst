How to document
===============

ODL is documented using sphinx_ and a `modified version of`_ numpydoc_. An example documentation is given below

.. code-block:: python
   
   class MyClass(object):
   
       """ Calculates important things 
       
       First line summarizes the class, after that comes a more
       detailed description
       """
     
       def __init__(self, c, parameter=None):
           """ initializer doc goes here
          
           Parameters
           ----------        
           c : `float`
               Constant to scale by
           parameter : `float`, optional
               Some extra parameter
           """
           self.c = c
           self.parameter = parameter
               
       def my_method(self, x, y):
           """ Calculate ``c * (x + y)``
           
           The first row is a summary, after that goes 
           a more detailed description.
           
           Parameters
           ----------
           x : `float`
               First summand
           y : `float`
               Second summand
               
           Returns
           -------
           scaled_sum : `float`
               Result of ``c * (x + y)``
               
           Examples
           --------
           Examples should be working pieces of code
           
           >>> obj = MyClass(5)
           >>> obj(3, 5)
           8.0
           """
           return self.c * (x + y)


Some short tips

* Writing withing backticks: ```some_target``` will create a link to the target.
* Make sure that the first line is short and descriptive.
* Examples are often better than long descriptions.
           
Advanced
--------
This is advanced topics for developers that need to change how the doc works.

Re-generating the doc
~~~~~~~~~~~~~~~~~~~~~

Autosummary currently does not support nestled modules, so to handle this we auto-generate rst files for each module. This is done using the ``doc/source/generate_doc.py`` script.

Modifications to numpydoc
~~~~~~~~~~~~~~~~~~~~~~~~~

Numpydoc has been modified in the following ways:

* The numpy shpinx domain has been removed.
* Added more ``extra_public_methods``
* autoclass summaries now link to full name, this allows subclassing between packages.



.. _sphinx: http://sphinx-doc.org/
.. _modified version of: https://github.com/odlgroup/numpydoc
.. _numpydoc: https://github.com/numpy/numpydoc