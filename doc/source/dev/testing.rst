##############
Testing in ODL
##############

ODL is tested using pytest_ and has four main types of tests and verification measures that are supposed to test the code in different ways. These are listed below along with the command to run them.

============  =====================  =======
Name          Command                Description
============  =====================  =======
Unittests     `pytest`               Test "micro-features" of the code, like testing that each parameter combination works, that the correct exceptions are raced and that the code works correctly in corner cases and with a wide range of input.
Largescale    `pytest --largescale`  Verify that the functionality works well in more complicated and realistic conditions.
Doctests      `pytest`               Meant as simple examples of how to use a functionality for users reading the documentation.
Examples      `pytest --examples`    Copy-paste friendly examples on how to use a function in a more complete context.
============  =====================  =======

Unittests
~~~~~~~~~
All unit-tests in ODL are contained in the `odl/tests` folder, where each ODL sub-package has a test file of its own. Any major ODL functionality should have unittests covering all of the use cases that are implied in the documentation. In addition to this, the tests should be quick to run, preferably at most a few milliseconds per tests. This is because the tests should be run as often as possible in order to verify that no functionality has been broken when adding new code or refactoring old code.

A short example of testing a function is given below, for more information consult the pytest documentation and look at existing tests in the test folder.

.. code:: python

    import numpy as np
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
            myfunction(np.inf)


Largescale
~~~~~~~~~~
Largescale test that functions work well even in realistic conditions and with a wide range of input, they live in `odl/tests/largescale`. Not all functionality needs largescale tests, in fact, most doesn't. However for some functions accurate tests even in realistic conditions are needed and these cannot be run very quickly.

It may also be the case that some functions accept a very large number of possible input configurations, in this case, testing the most common configuration in the regular unittest and testing the others in a largescale test is acceptable.

Doctests
~~~~~~~~
Doctests are the most simple type of test used in ODL, and are mainly meant as examples on how to use functionality. They can be included by using the Examples header in an usual docstring, as shown below:

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

Dispite simply looking like documentation, doctests are actual pieces of python code and will be executed when the ``pytest`` command is invoked. All ODL source files also contain the lines

.. code:: python

    if __name__ == '__main__':
        from odl.util.testutils import run_doctests
        run_doctests()

which mean that if a ODL source file is executed in isolation, all the doctests in the file are run. This can be useful during development in order to quickly see if some functionality works as expected.

Examples
~~~~~~~~
Examples, while not technically tests in the traditional sense, still constitute part of the test framework for ODL by showing how different parts of ODL work togeather and by ensuring that functions that depend on each other work as expected.

It is even possible to run all examples as part of the test suite by running ``pytest --examples``, but be aware that this requires all ODL dependencies to be installed.

For examples on how to write examples, please consult the examples directory.

.. _pytest: http://doc.pytest.org/en/latest/