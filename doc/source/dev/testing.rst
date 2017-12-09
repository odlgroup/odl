.. _dev_testing:

##############
Testing in ODL
##############

ODL tests are run using pytest_, and there are several types:


==============  =========================  =======
Name            Command                    Description
==============  =========================  =======
Unit tests      ``pytest``                 Test "micro-features" of the code
Large-scale     ``pytest --largescale``    Unit tests with large inputs and more cases
Doctests        ``pytest``                 Validate usage examples in docstrings
Examples        ``pytest --examples``      Run all examples in the `examples`_ folder
Documentation   ``pytest --doctest-doc``   Run the doctest examples in the Sphinx documentation
==============  =========================  =======

Unit tests
~~~~~~~~~~
All unit tests in ODL are contained in the `test`_ folder, where each sub-package has a test file on its own.
Any major ODL functionality should have unit tests covering all of the use cases that are implied in the documentation.
In addition to this, the tests should be quick to run, preferably at most a few milliseconds per test.
If the test suite takes too long to run, users and developers won't run them as often as necessary to make sure that they didn't break any functionality.

A short example of testing a function is given below.
For more information consult the `pytest`_ documentation and look at existing tests in the `test`_ folder.

.. code:: python

    import pytest


    def myfunction(x):
        """Convert ``x`` to a integer and add 1"""
        return int(x) + 1


    def test_myfunction():
        # Test basic functionality
        assert myfunction(1) == 2
        assert myfunction(-3) == -2
        assert myfunction(10) == 11

        # Test when called with float
        assert myfunction(1.5) == 2

        # Test when called with string
        assert myfunction('1') == 2

        # Verify that bad input throws a proper error
        with pytest.raises(TypeError):
            myfunction([])

        with pytest.raises(ValueError):
            myfunction('non-integer')

        with pytest.raises(TypeError):
            myfunction(object())

        with pytest.raises(OverflowError):
            myfunction(float('inf'))


Large-scale
~~~~~~~~~~~
Large-scale test verify that functions work well even in realistic conditions and with an even wider range of input than in the standard unit tests.
They live in the ``largescale`` subfolder of the `test`_ folder.
Not all functionality needs largescale tests, in fact, most doesn't.
This type of test makes most sense for (1) functionality that has a complex implementation where it's easy to make mistakes that the code slow (regression tests) and (2) features that take too much time to be tested broadly in the standard suite.
For the second type, the unit tests should include only a couple of tests that can run fast, and the full range of inputs can be tested in the large-scale suite.

It may also be the case that some functions accept a very large number of possible input configurations, in this case, testing the most common configuration in the regular unittest and testing the others in a largescale test is acceptable.

Doctests
~~~~~~~~
Doctests are the simplest type of test used in ODL, and are snippets of code that document the usage of functions and classes and can be run as small tests at the same time.
They can be included by using the Examples header in an usual docstring, as shown below:

.. code:: python

    def myfunction(x):
        """Convert ``x`` to a integer and add 1.

        Examples
        --------
        For integers, the function simply adds 1:

        >>> myfunction(1)
        2

        The function also works with floats:

        >>> myfunction(1.3)
        2
        """
        return int(x) + 1

Despite simply looking like documentation, doctests are actual pieces of python code and will be executed when the ``pytest`` command is invoked.
See the `doctest` documentation for more information.

All ODL source files should also contain the lines:

.. code:: python

    if __name__ == '__main__':
        from odl.util.testutils import run_doctests
        run_doctests()

which mean that if a ODL source file is executed in isolation, all the doctests in the file are run.
This can be useful during development in order to quickly see if some functionality works as expected.

Examples
~~~~~~~~
Examples, while not technically tests in the traditional sense, still constitute a part of the test framework for ODL by showing how different parts of ODL work together and by ensuring that functions that depend on each other work as expected.
The main purpose of the examples is however to show ODL from a users perspective and particular care should be taken to keep them readable and working since this is often the first thing users see when they start using ODL.

It is even possible to run all examples as part of the test suite by running ``pytest --examples``, but be aware that this requires all ODL dependencies to be installed and that plotting windows can be opened during execution.

Consult the `examples`_ directory for an impression of the style in which ODL examples are written.

.. _doctest: https://docs.python.org/library/doctest.html
.. _pytest: http://doc.pytest.org/en/latest/
.. _examples: https://github.com/odlgroup/odl/tree/master/examples
.. _test: https://github.com/odlgroup/odl/tree/master/odl/test
