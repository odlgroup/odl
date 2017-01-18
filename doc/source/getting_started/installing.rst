.. _installing_odl:

##############
Installing ODL
##############

This guide will go through all steps necessary for a full ODL installation, starting from nothing more than a working operating system (Linux, MacOS or Windows).

************
Introduction
************

Installing ODL is intended to be straightforward, and this guide is meant for new users.
For a working installation you should perform the following steps:

1. Install a Python interpreter
2. Install ODL and its dependencies
3. (optional) Install extensions for more functionality
4. (optional) Run the tests

Consider using Anaconda
=======================
We currently recommend to use `Anaconda`_ on all platforms since it offers the best out-of-the-box installation and run-time experience.
Anaconda also has other benefits, for example the possibility to work in completely isolated Python environments with own installed packages, thereby avoiding conflicts with system-wide installed packages.
Furthermore, Anaconda cooperates with ``pip`` (see below), i.e. packages can be installed with both Anaconda's internal mechanism and ``pip`` without conflicts.

Alternatively, packages can be installed with `pip`_ in a user's location, which should also avoid conflicts.
We will provide instructions for this alternative.

Another possibility is to use `virtualenv`_, which can be seen as a predecessor to Anaconda.
Following the ``pip`` installation instructions in a ``virtualenv`` without the ``--user`` option works very well in our experience, but we do not provide explicit instructions for this variant.

Which Python version to use?
============================
Any modern Python distribution supporting `NumPy`_ and `SciPy`_ should work for the core library, but some extensions require CPython (the standard Python distribution).

ODL fully supports both Python 2 and Python 3.
If you choose to use your system Python interpreter (the "pip install as user" variant), it may be a good idea to stick with the default one, i.e. the one invoked by ``python``.
Otherwise, we recommend using Python 3, since Python 2 support will be discontinued in 2020.

Development enviroment
======================
Since ODL is object-oriented, using an Integrated Development Environment (IDE) is recommended, but not required.
The most popular ones are `Spyder`_ which works on all major platforms and can be installed through both ``conda`` and ``pip``, and `PyCharm`_ which can be integrated with any text editor of your choice, such as Emacs or Vim.


------

.. _alternative_1:

***************************************************************************
Alternative 1: Installing a release using ``conda`` (recommended for users)
***************************************************************************

TL;DR
=====
Instructions for the impatient:

- Download and install `Miniconda`_
- Create conda environment:

  .. code-block:: bash

    $ conda create -c odlgroup -n odl-py35 python=3.5 odl matplotlib pytest scikit-image spyder

- Activate the conda enviroment and start working!


.. _installing_anaconda:

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
    If you want to use `Spyder`_ as integrated development environment (IDE, see `Development enviroment`_) on Linux or MacOS, you should also install it in the new conda environment and run it from there.
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

Extra dependencies
------------------
The following packages are optional and extend the functionality of ODL.

- Image and plot displaying capabilities:

  .. code-block:: bash

    $ conda install matplotlib

- Faster FFT back-end using FFTW (currently not in mainstream conda):

  * Install the `FFTW`_ C library version 3 (all possible precisions).
    Use your Linux package manager for this task or consult the `Windows <http://fftw.org/install/windows.html>`_ or `MacOS <fftw.org/install/mac.html>`_ instructions, respectively.

  * Run

    .. code-block:: bash

        $ pip install pyfftw

- Wavelet transforms (currently not in mainstream conda):

  .. code-block:: bash

    $ pip install pywavelets

- Simple backend for ray transforms:

  .. code-block:: bash

    $ conda install scikit-image

- Fast `ASTRA`_ ray transform backend:

  .. code-block:: bash

    $ conda install -c astra-toolbox astra-toolbox

  If this doesn't work, or if you want a more recent version, check out the `ASTRA for X-ray tomography`_ section below.

- Bindings to the `ProxImaL`_ convex optimization package, an extension of `CVXPY`_:

  .. code-block:: bash

    $ pip install proximal

- To run unit tests:

  .. code-block:: bash

    $ conda install pytest


--------

.. _alternative_2:

*************************************************
Alternative 2: Installing a release using ``pip``
*************************************************

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

.. warning::

    Correctly installing ODL's dependencies on Windows, especially Numpy and Scipy, can be quite a hassle, and we therefore discourage this variant.
    You should really consider using Anaconda instead, see :ref:`alternative_1`.


Installing ODL and its dependencies
===================================
You may need to `install pip`_ to be able to install ODL and its dependencies from the `Python Package Index`_ (PyPI).
If running ``pip`` (alternatively: ``pip2`` or ``pip3``) shows a help message, it is installed -- otherwise you need to install it first.

For basic installation without extra dependencies, run

.. code-block:: bash

   $ pip install --user odl


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


------

.. _alternative_3:

********************************************************************
Alternative 3: Installation from source (recommended for developers)
********************************************************************
This installation method is intended for developers who want to make changes to the code.
It assumes that the `Git`_ version control system is available on your system; for up-to-date instructions, consult the `Git installation instructions <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.
You also need `pip`_ to perform the installation.

.. note::
    You should consider performing all described steps in a `conda environment <http://conda.pydata.org/docs/using/envs.html>`_ -- it gives you the same encapsulation benefits as developer that you would enjoy also as a user (no conflicting packages, free to choose Python version, ...).
    See the `Installing Anaconda`_ section for setup instructions.

To get ODL, navigate to a folder where you want the ODL repository to be stored and clone the repository with the command

.. code-block:: bash

   $ git clone https://github.com/odlgroup/odl

No GitHub account is required for this step.


In a conda environment
======================
This part assumes that you have activated a conda environment before (see :ref:`installing_anaconda`).

You can choose to install dependencies first:

**On Linux/MacOS:**

.. code-block:: bash

    $ conda install nomkl numpy scipy future matplotlib

**On Windows:**

.. code-block:: bash

    $ conda install numpy scipy future matplotlib

After that, enter the top-level directory of the cloned repository and run

.. code-block:: bash

   $ pip install --editable .
   
**Optional dependencies:**

You may also want to install optional dependencies:

.. code-block:: bash

    $ conda install matplotlib pytest pytest-pep8

Using only ``pip``
==================
Enter the top-level directory of the cloned repository and run

.. code-block:: bash

   $ pip install --user --editable .


.. warning::
    **Don't forget the "." (dot) at the end** - it refers to the current directory, the location from where ``pip`` is supposed to install ODL.

.. note::
    We recommend the ``--editable`` option (can be shortened to ``-e``) since it installs a link instead of copying the files to your Python packages location.
    This way, local changes to the code (e.g. after a ``git pull``) take immediate effect after reloading the package, without requiring reinstallation.


Further developer information
=============================
See :ref:`Contributing to ODL <contributing>` for more information.


------

.. _running_the_tests:

****************
Runing the tests
****************
Unit tests in ODL are based on `pytest`_.
They can be run either from within ``odl`` or by invoking ``pytest`` directly.

Installing testing dependencies
===============================
If you installed an ODL release using ``conda`` or ``pip``, respectively, you should install ``pytest`` using the same method.
For source installations, you can choose your favorite method below.

Using ``conda``:
----------------
If you installed ODL using conda, ``pytest`` is already installed as dependency, so there should not be anything left to do.
Otherwise, you can install it by running

.. code-block:: bash

    $ conda install pytest

Using ``pip``:
--------------
.. code-block:: bash

    $ pip install --user odl[testing]


Testing the code
================
Now you can check that everything was installed properly by running

.. code-block:: bash

   $ python -c "import odl; odl.test()"

.. note::
    If you have several versions of ODL and run this command in the top-level directory of an ODL clone, the tests in the repository will be run, not the ones in the installed package.

If you have installed ODL from source, you can also use ``pytest`` directly in the root of your ODL clone:

.. code-block:: bash

   $ pytest

------


*******************
Compiled extensions
*******************
There are several compiled extensions to ODL.
Some of them can be installed using ``conda``, others require manual compilation.


CUDA backend for linear arrays
==============================
The `odlcuda`_ backend for fast array calculations on CUDA requires the `CUDA toolkit`_ (on Linux: use your distro package manager) and a CUDA capable graphics card with compute capability of at least 3.0.
Search `this table <https://en.wikipedia.org/wiki/CUDA#GPUs_supported>`_ for your model.

Installation using ``conda``
----------------------------
.. note::
    In conda, the ``odlcuda`` package is currently available only for Linux 64-bit and Python 3.5.
    Furthermore, you may experience failures due to "invalid device function" -- this is a known issue, and we're trying to fix it.

If you have installed an ODL release, simply run (in a directory of your choice)

.. code-block:: bash

    $ conda install -c odlgroup odlcuda

If you have installed ODL from source, you need to prevent conda from installing its version of ODL.
To do this, find out the dependencies of ``odlcuda`` by running

.. code-block:: bash

    $ conda install --dry-run odlcuda

Install all its dependencies except ``odl`` and ``odlcuda``.
Finally, install ``odlcuda`` without dependencies:

.. code-block:: bash

    $ conda install --no-deps odlcuda

Building from source
--------------------
You have two options of building ``odlcuda`` from source.
For both, first clone the ``odlcuda`` GitHub repository and enter the new directory:

.. code-block:: bash

    $ git clone https://github.com/odlgroup/odlcuda.git
    $ cd odlcuda

1. **Using conda build**

   This is the simpler option and should work on any Linux or MacOS system (we currently have no Windows build recipe, sorry).

   To build the conda recipe, you should be **in the root conda environment** (see :ref:`installing_anaconda` for details) and in the top-level directory of your ``odlcuda`` clone.
   You also need the ``conda-build`` package, which is installed by

   .. code-block:: bash

       $ conda install conda-build

   Next, switch to the ``conda-build`` branch:

   .. code-block:: bash

       $ git checkout conda-build

   Finally, build the package using ``conda build``.
   Currently, this requires you to manually provide the location of the CUDA toolkit and the compute capability of your graphics card using the environment variables ``CUDA_ROOT`` and ``CUDA_COMPUTE``.
   (If you forget them, the build recipe will only issue a warning in the beginning but fail later on.)
   The ``CUDA_ROOT`` is given as path, e.g. ``/usr/local/cuda``, and ``CUDA_COMPUTE`` as 2-digit number without dot, e.g. ``30``.

   .. note::
       You can consult `this table <https://en.wikipedia.org/wiki/CUDA#GPUs_supported>`_ for the compute capability of your device.
       The minimum required is ``30``, which corresponds to the "Kepler" generation.

   Assuming the example configuration above, the build command to run is

   .. code-block:: bash

       $ CUDA_ROOT=/usr/local/cuda CUDA_COMPUTE=30 conda build ./conda

   This command builds ``odlcuda`` in a separate build conda environment and tries to import it and run some tests after the build has finished.
   If all goes well, you will get a message at the end that shows the path to the conda package.

   Finally, install this package file **in your working conda environment** (e.g. ``source activate odl-py35``) by invoking e.g.

   .. code-block:: bash

       $ conda install /path/to/your/miniconda/conda-bld/linux-64/odlcuda-0.3.0-py35_0.tar.bz2


2. **Manually with CMake**

   This option requires more manual work but is known to work on all platforms.

   See `here <https://github.com/odlgroup/odlcuda.git>`_ for build instructions.
   You may want to use include and library paths (GCC, boost, ...) of a conda enviroment and install the package in it.

A simple test if this build of ``odlcuda`` works, you can run

.. code-block:: bash

    $ python -c "import odl; odl.rn(3, impl='cuda').element()"

If you get a ``KeyError: 'cuda'``, then something went wrong with the package installation since it cannot be imported.
If the above command instead raises a ``MemoryError`` or similar, your graphics card is not properly configured, and you should solve that issue first.


ASTRA for X-ray tomography
==========================
To calculate fast forward and backward projections for image reconstruction in X-ray tomography, install the `ASTRA tomography toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
ASTRA projectors are fully supported in ODL.

You can try using the conda package, but we can give no guarantee that it works out of the box:

.. code-block:: bash

    $ conda install -c astra-toolbox astra-toolbox

For further instructions, check `the ASTRA GitHub page <https://github.com/astra-toolbox/astra-toolbox>`_.


STIR for emission tomography
============================
For applications in emission tomography, i.e. PET or SPECT, install `STIR`_ with Python bindings.
Support for STIR is currently very limited.


******
Issues
******
If you have any problems during installation, consult the help in the :ref:`FAQ <FAQ>`.
If that does not help, `make an issue on GitHub <https://github.com/odlgroup/odl/issues>`_ or send us an email (odl@math.kth.se) and we'll try to assist you promptly.


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
