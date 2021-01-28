.. _installing_odl_conda:

##########################
Installing ODL using conda
##########################

Anaconda is a binary distribution package that allows user to install pre-compiled python packages in a very simple manner.
It works on all platforms and is the recommended way of installing ODL as a user.
If you already have anaconda installed, you can go directly to `Installing ODL and its dependencies`_, otherwise you need to begin by installing anaconda.


.. _installing_odl_conda__tldr:

TL;DR
=====
Instructions for the impatient:

- Download and install `Miniconda`_
- Create conda environment:

  .. code-block:: bash

    $ conda create -c odlgroup -n odl-py35 python=3.5 odl matplotlib pytest scikit-image spyder

- Activate the conda enviroment and start working!


.. _installing_odl_conda__installing_anaconda:

Installing Anaconda
===================
Even though a Python interpreter is included by default in virtually all Linux distributions, it is advisable to use Anaconda's Python ecosystem since it gives you full flexibility in the Python version you use and which packages you install.

Download Anaconda from the Continuum Analytics home page.
You may choose to download the `full Anaconda <https://www.continuum.io/downloads>`_ variant, but we recommend the slim `Miniconda`_ distribution since many of the packages included in full Anaconda are out of date anyway and need to be updated.
Note that the choice of Python version (2 vs. 3) of the Anaconda installer is not very important since you can later choose to create conda environments with any Python version (see below).

Make sure that during installation, your ``PATH`` variable is adapted such that ``conda`` and other scripts can be found by your shell::

    Do you wish the installer to prepend the Miniconda3 install location
    to PATH in your /home/user/.bashrc ? [yes|no]
    [no] >>> yes

After restarting the terminal (for the changed ``PATH`` variable to take effect), you can run

.. code-block:: bash

   $ conda update --all

to make sure you have the latest versions of all packages.

Optionally, create a new conda environment to work with ODL.
This is a very convenient way to have several "ecosystems" of Python packages in parallel without mutual interference:

.. code-block:: bash

    $ conda create --name odl-py35 python=3.5

Enter the newly created conda environment by running ``source activate odl-py35`` (Linux/MacOS) or ``activate odl-py35`` (Windows).
If you want to exit later on, run ``source deactivate`` (Linux/MacOS) or ``deactivate`` (Windows), respectively.
See the `Managing conda environments`_ documentation for further information.

.. note::
    If you want to use `Spyder`_ as integrated development environment (IDE, see :ref:`installing_odl__development_environment`) on Linux or MacOS, you should also install it in the new conda environment and run it from there.
    Otherwise, Spyder may not able to use the packages in the conda environment:

    .. code-block:: bash

        $ conda install spyder

    On Windows, you can install Spyder in the root conda environment (run ``deactivate`` to get there), but you need to change its default Python interpreter.
    To do this, open Spyder and use the navigation bar to open "Tools -> Preferences".
    Click on "Python interpreter" and change the first setting "Select the Python interpreter for all Spyder consoles" from the default setting to "Use the following Python interpreter:".
    In the text field, fill in the path to the Python executable in your newly created conda environment.
    For example, if you installed Miniconda (or Anaconda) in ``C:\Programs\Miniconda3``, then the environment's Python interpreter is ``C:\Programs\Miniconda3\envs\odl-py35\bin\python.exe``.
    You can use the file system browser (symbol to the right of the text field) to find the interpreter on your system.


Installing ODL and its dependencies
===================================
Install ODL and all its (minimal) dependencies in a ``conda`` environment of your choice by running

.. code-block:: bash

    $ conda install -c odlgroup odl

.. note::
    To skip the ``-c odlgroup`` option in the future, you can permanently add the ``odlgroup`` conda channel (see `Managing conda channels`_):

    .. code-block:: bash

        $ conda config --append channels odlgroup

    After that, ``conda install odl`` and ``conda update odl`` work without the ``-c`` option.


.. _installing_odl_conda__extensions:

Extra dependencies
------------------
The following packages are optional and extend the functionality of ODL.
Some of them require `pip`_ in order to be installed. See `install pip`_ for
further instructions.

- Image and plot displaying capabilities using `matplotlib`_:

  .. code-block:: bash

    $ conda install matplotlib

- Faster FFT back-end using FFTW (currently not in mainstream conda):

  * Install the `FFTW`_ C library version 3 (all possible precisions).
    Use your Linux package manager for this task or consult the `Windows <http://fftw.org/install/windows.html>`_ or `MacOS <fftw.org/install/mac.html>`_ instructions, respectively.

  * Install the python backend `pyFFTW`_ by running:

    .. code-block:: bash

        $ pip install pyfftw

- Wavelet transforms (currently not in mainstream conda) using `PyWavelets`_:

  .. code-block:: bash

    $ pip install pywavelets

- Simple backend for ray transforms using `scikit-image`_:

  .. code-block:: bash

    $ conda install scikit-image

- Fast `ASTRA`_ ray transform backend:

  .. code-block:: bash

    $ conda install -c astra-toolbox astra-toolbox

  If this doesn't work, or if you want a more recent version, see `the ASTRA GitHub page <https://github.com/astra-toolbox/astra-toolbox>`_.

- Bindings to the `ProxImaL`_ convex optimization package, an extension of `CVXPY`_:

  .. code-block:: bash

    $ pip install proximal

More information can be found in :ref:`installing_odl_extensions`.


.. _installing_odl_conda__running tests:

Running the tests
=================
Unit tests in ODL are based on `pytest`_.
To run the tests, you first need to install the testing framework:

.. code-block:: bash

    $ conda install pytest

Now you can check that everything was installed properly by running

.. code-block:: bash

   $ python -c "import odl; odl.test()"

.. note::
    If you have several versions of ODL and run this command in the top-level directory of an ODL clone, the tests in the repository will be run, not the ones in the installed package.


.. _Anaconda: https://anaconda.org/
.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Managing conda environments: http://conda.pydata.org/docs/using/envs.html
.. _Managing conda channels: http://conda.pydata.org/docs/channels.html

.. _pip: https://pip.pypa.io/en/stable/
.. _install pip: https://pip.pypa.io/en/stable/installing/#installation

.. _Spyder: https://github.com/spyder-ide/spyder

.. _pytest: https://pypi.python.org/pypi/pytest

.. _matplotlib: http://matplotlib.org/
.. _FFTW: http://fftw.org/
.. _pyFFTW: https://pypi.python.org/pypi/pyFFTW
.. _PyWavelets: https://pypi.python.org/pypi/PyWavelets
.. _scikit-image: http://scikit-image.org/
.. _ProxImaL: http://www.proximal-lang.org/en/latest/
.. _CVXPY: http://www.cvxpy.org/en/latest/
.. _ASTRA: https://github.com/astra-toolbox/astra-toolbox
