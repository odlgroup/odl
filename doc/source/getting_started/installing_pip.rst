.. _installing_odl_pip:

========================
Installing ODL using pip
========================
`pip`_ is a package manager that works on all major platforms and allows user to install python packages in a very simple manner.
If you already have python and pip installed, you can go directly to `Installing ODL and its dependencies`_, otherwise you need to begin by installing python and pip.

.. warning::

    Correctly installing ODL's dependencies on Windows, especially Numpy and other compiled dependencies, can be quite a hassle, and we therefore discourage this variant.
    You should really consider :ref:`using Anaconda instead <installing_odl_conda>`.
    
TL;DR
=====
Instructions for the impatient:

- Install `pip`_
- Install ODL and dependencies:

  .. code-block:: bash

    $ pip install odl[show,pywavelets,scikit,proximal,testing]

Installing a Python interpreter
===============================
Open a terminal and type ``python`` + Enter.
If a Python prompt appears, you already have an interpreter installed and can skip this step (exit by running ``exit()``).
Otherwise, you need to install it.

On Linux:
---------
In the unlikely event that Python is not installed, consult your distro package manager.

On MacOS:
---------
Get the latest release (2 or 3) for MacOS `here <https://www.python.org/downloads/mac-osx/>`_ and install it.

On Windows:
-----------
Python installers can be downloaded from `this link <https://www.python.org/downloads/windows/>`_.
Pick the latest release for your favorite version (2 or 3).


Installing ODL and its dependencies
===================================
You may need to `install pip`_ to be able to install ODL and its dependencies from the `Python Package Index`_ (PyPI).
If running ``pip`` (alternatively: ``pip2`` or ``pip3``) shows a help message, it is installed -- otherwise you need to install it first.

For basic installation without extra dependencies, run

.. code-block:: bash

   $ pip install --user odl

.. _installing_odl_pip_extensions:
   
Extra dependencies
------------------
The following optional packages extend the functionality of ODL.
They can be specified as keywords in square brackets, separated by commas (no spaces!):

.. code-block:: bash

   $ pip install odl[dep1,dep2]

Possible choices:

- ``show`` : Install matplotlib_ to enable displaying capabilities.
- ``fft`` : Install `pyFFTW`_ for fast Fourier transforms. Note that this requires the `FFTW`_ C library to be available on your system.
  Note also that even without this dependency, FFTs can be computed with Numpy's FFT library.
- ``pywavelets`` : Install `PyWavelets`_ for wavelet transforms.
- ``scikit`` : Install `scikit-image`_ as a simple backend for ray transforms.
- ``proximal``: Install the `ProxImaL`_ convex optimization package.
- ``testing``: Pull in the dependencies for unit tests (see :ref:`running_the_tests`)

These dependencies are optional and may not be easy to install on your system (especially on Windows).
In general, a clean ODL installation is enough for most users' initial needs.

More information can be found in :ref:`installing_odl_extensions`.

Running the tests
=================
Unit tests in ODL are based on `pytest`_.
To run the tests, you first need to install the testing framework:

.. code-block:: bash

    $ pip install --user odl[testing]

Now you can check that everything was installed properly by running

.. code-block:: bash

   $ python -c "import odl; odl.test()"
   
.. note::
    If you have several versions of ODL and run this command in the top-level directory of an ODL clone, the tests in the repository will be run, not the ones in the installed package.
    
    
.. _Anaconda: https://anaconda.org/
.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Managing conda environments: http://conda.pydata.org/docs/using/envs.html
.. _Managing conda channels: http://conda.pydata.org/docs/channels.html

.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _pip: https://pip.pypa.io/en/stable/
.. _install pip: https://pip.pypa.io/en/stable/installing/#installation
.. _Python Package Index: https://pypi.python.org/pypi

.. _Spyder: https://github.com/spyder-ide/spyder
.. _PyCharm: https://www.jetbrains.com/pycharm/

.. _Git: http://www.git-scm.com/
.. _msysgit: http://code.google.com/p/msysgit/downloads/list
.. _git-osx-installer: http://code.google.com/p/git-osx-installer/downloads/list
.. _GitHub Help : https://help.github.com/

.. _pytest: https://pypi.python.org/pypi/pytest
.. _coverage: https://pypi.python.org/pypi/coverage/

.. _NumPy: http://www.numpy.org/
.. _SciPy: https://www.scipy.org/
.. _future: https://pypi.python.org/pypi/future/
.. _matplotlib: http://matplotlib.org/
.. _FFTW: http://fftw.org/
.. _pyFFTW: https://pypi.python.org/pypi/pyFFTW
.. _FFTW: http://fftw.org/
.. _PyWavelets: https://pypi.python.org/pypi/PyWavelets
.. _scikit-image: http://scikit-image.org/
.. _ProxImaL: http://www.proximal-lang.org/en/latest/
.. _CVXPY: http://www.cvxpy.org/en/latest/
.. _odlcuda: https://github.com/odlgroup/odlcuda
.. _CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
.. _ASTRA: https://github.com/astra-toolbox/astra-toolbox
.. _STIR: https://github.com/UCL/STIR
