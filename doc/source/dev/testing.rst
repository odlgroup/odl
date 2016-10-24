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
All unit-tests in ODL are contained in the `odl/tests` folder, where each ODL sub-package has a test file of its own. Any major ODL functionality should have unittests covering all of the use cases that are implied in the documentation. In addition to this, the tests should be quick to run, preferably at most a few milliseconds per tests. This is because the tests should be run as often as possible in order to verify that no functionality has been broken.

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

Doctests
~~~~~~~~

Examples
~~~~~~~~



.. _pytest: http://doc.pytest.org/en/latest/