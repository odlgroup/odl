.. _installing_odl:

##############
Installing ODL
##############

Introduction
============

Installing ODL is intended to be straightforward, and this guide is meant for new users.
For a minimal installation you should perform the following steps:

1. Install a Python interpreter
2. Install ODL and its dependencies
3. (optional) Run the tests

Consider using Anaconda
-----------------------
We currently recomment to use `Anaconda`_ on all platforms since it offers the best out-of-the-box installation and run-time experience.
Anaconda also has other benefits, for example the possibility to work in completely isolated Python environments with own installed packages, thereby avoiding conflicts with system-wide installed packages.
Furthermore, Anaconda cooperates with ``pip`` (see below), i.e. packages can be installed with both Anaconda's internal mechanism and ``pip`` without conflicts.

Alternatively, packages can be installed with `pip`_ in a user's location, which should also avoid conflicts.
We will provide instructions for this alternative.

Another possibility is to use `virtualenv`_, which can be seen as a predecessor to Anaconda.
Following the ``pip`` installation instructions in a ``virtualenv`` without the ``--user`` option should work, but we do not provide explicit instructions for this variant.

Which Python version to use?
----------------------------
Any modern Python distribution supporting `NumPy`_ and `SciPy`_ should work for the core library, but some extensions require CPython (the standard Python distribution).

ODL fully supports both Python 2 and Python 3.
If you choose to use your system Python interpreter (the "pip install as user" variant), it may be a good idea to stick with the default one, i.e. the one invoked by ``python``.
Otherwise, we recommend using Python 3, since Python 2 support will be discontinued in 2020.

Development enviroment
----------------------
Since ODL is object-oriented, using an Integrated Development Environment (IDE) is recommended, but not required.
The most popular ones are `Spyder`_ which works on all major platforms and can be installed through both ``conda`` and ``pip``, and `PyCharm`_ which can be integrated with any text editor of your choice, such as Emacs or Vim.


Alternative 1: Installing a release using ``conda`` (recommended for users)
===========================================================================

TL;DR
-----
- Download and install `Miniconda`_
- Create conda environment:

  .. code-block:: bash

    $ conda create -c odlgroup -n odl-py35 python=3.5 odl odlcuda matplotlib pytest scikit-image spyder

Install Anaconda
----------------
Even though a Python interpreter is included by default in virtually all Linux distributions, it is advisable to use Anaconda's Python  ecosystem since it gives you full flexibility in the Python version you use and which packages you install.

Download Anaconda from the Continuum Analytics home page.
You may choose to download the `full Anaconda <https://www.continuum.io/downloads>`_ variant, but we recommend the slim `Miniconda`_ distribution since many of the packages included in full Anaconda are out of date anyway and need to be updated.
Note that the choice of Python version (2 vs. 3) of the Anaconda installer is not very important since you can later choose to create conda environments with any Python version (see below).

Make sure that during installation, your ``PATH`` variable is adapted such that ``conda`` and other scripts can be found by your shell::

    Do you wish the installer to prepend the Miniconda3 install location
    to PATH in your /home/user/.bashrc ? [yes|no]
    [no] >>> yes

After opening a new terminal (for the changed ``PATH`` variable to take effect), you can run

.. code-block:: bash

   $ conda update --all

to make sure you have the latest versions of all packages.

Optionally, create a new conda environment to work with ODL.
This is a very convenient way to have several "ecosystems" of Python packages in parallel without mutual interference:

.. code-block:: bash

    $ conda create --name odl-py35 python=3.5

Follow the displayed instructions to enter/exit the newly created conda environment with name ``odl-py35``.
See the `Managing conda environments`_ documentation for further information.

If you use `Spyder`_ as integrated development environment (IDE, see `Development enviroment`_), you should also install it in the new conda environment and run it from there.
Otherwise, Spyder is not able to use the packages in the conda environment:

.. code-block:: bash

    $ conda install spyder


Install ODL and its dependencies
--------------------------------
Install ODL and all its (minimal) dependencies by running

.. code-block:: bash

    $ conda install -c odlgroup odl

To skip the ``-c odlgroup`` option in the future, you can permanently add the ``odlgroup`` conda channel (see `Managing conda channels`_):

.. code-block:: bash

    $ conda config --append channels odlgroup

After that, ``conda install odl`` and ``conda update odl`` work without the ``-c`` option.

**Extra dependencies**

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

- To run unit tests:

  .. code-block:: bash

    $ conda install pytest




Alternative 2: Installing a release using ``pip``
=================================================

TL;DR
-----
- Install `pip`_
- Install `FFTW`_
- Install ODL and dependencies:

  .. code-block:: bash

    $ pip install odl[show,fftw,pywavelets,scikit,testing]

Install a Python interpreter
----------------------------
Open a terminal and type ``python`` + Enter.
If a Python prompt appears, you already have an interpreter installed and can skip this step.
Otherwise, you need to install it.

**On Linux:**

In the unlikely event that Python is not installed, consult your distro package manager.

**On MacOS:**

Get the latest release (2 or 3) for MacOS `here <https://www.python.org/downloads/mac-osx/>`_ and install it.

**On Windows:**

Installers can be downloaded from `this link <https://www.python.org/downloads/windows/>`_.
You should pick the latest release for your favorite version (2 or 3).


Install ODL and its dependencies
--------------------------------
You need to `install pip`_ to be able to install ODL and its dependencies from the `Python Package Index`_ (PyPI).
Check if running ``pip`` (alternatively: ``pip3``) shows a help message or an error.

For basic installation without extra dependencies, run

.. code-block:: bash

   $ pip install --user odl


**Extra dependencies**

Additional dependencies can be specified in square brackets, separated by commas (no spaces!):

.. code-block:: bash

   $ pip install odl[dep1,dep2]

Possible choices:

- ``all`` : Install with all extra dependencies.
- ``show`` : Install matplotlib_ to enable displaying capabilities.
- ``fft`` : Install `pyFFTW`_ for fast Fourier transforms. Note that this requires the `FFTW`_ C library to be available on your system.
  Note also that even without this dependency, FFTs can be computed with Numpy's FFT library.
- ``pywavelets`` : Install `PyWavelets`_ for wavelet transforms.
- ``scikit`` : Install `scikit-image`_ as a simple backend for ray transforms.
- ``proximal``: Install the `ProxImaL`_ convex optimization package.
  This package currently only works with Python 2 since it depends on OpenCV whose Python bindings have not been ported to version 3 yet.


These dependencies are optional and may not be easy to install on your system (especially on Windows).
In general, a clean ODL installation is enough for most users' initial needs.


Alternative 3: Installation from source (recommended for developers)
====================================================================
This installation method is intended for developers who want to make changes to the code.
It assumes that the `Git`_ version control system is available on your system; for up-to-date instructions, consult the `Git installation instructions <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.
You also need `pip`_ to perform the installation.

You should also consider performing all described steps in a `conda environment <http://conda.pydata.org/docs/using/envs.html>`_ -- it gives you the same encapsulation benefits as developer that you would enjoy also as a user (no conflicting packages, free to choose Python version, ...).
See the `Install Anaconda`_ section for setup instructions.

To get ODL, clone the repository with the command

.. code-block:: bash

   $ git clone https://github.com/odlgroup/odl

No GitHub account is required for this step.


**In a conda environment:**

You can choose to install dependencies first (optional ones in square brackets):

.. code-block:: bash

    $ conda install nomkl numpy scipy future [matplotlib]

After that, enter the top-level directory of the cloned repository and run

.. code-block:: bash

   $ pip install --editable .

**If you use pip only:**

Enter the top-level directory of the cloned repository and run

.. code-block:: bash

   $ pip install --user --editable .


**Don't forget the "." (dot) at the end** - it refers to the current directory, the location from where ``pip`` is supposed to install ODL.

We recommend the ``--editable`` option (can be shortened to ``-e``) since it installs a link instead of copying the files to your Python packages location.
This way, local changes to the code (e.g. after a ``git pull``) take immediate effect without reinstallation.


Further developer information
-----------------------------
See :ref:`Contributing to ODL <contributing>` for more information.


Runing the tests
================
Unit tests in ODL are based on `pytest`_, and coverage reports are created by the `coverage`_ module.

**Using conda:**

.. code-block:: bash

    $ conda install pytest pytest-cov pytest-pep8

**Using pip:**

.. code-block:: bash

    $ pip install --user pytest pytest-cov pytest-pep8

Now you can check that everything was installed properly by running

.. code-block:: bash

   $ python -c "import odl; odl.test()"

**anywhere but** in the top-level directory of an ODL clone.
If you have installed ODL from source, you can also use ``pytest`` directly:

.. code-block:: bash

   $ pytest



Compiled extensions
===================
There are several compiled extensions to ODL.
Some of them can be installed using conda, others require manual compilation.


CUDA backend for linear arrays
------------------------------
The `odlcuda`_ backend for fast array calculations on CUDA requires the `CUDA toolkit`_ (on Linux: use your distro package manager) and a CUDA capable graphics card with compute capability of at least 5.0.
Search `this table <https://en.wikipedia.org/wiki/CUDA#GPUs_supported>`_ for your model.

**Using conda:**

The custom CUDA backend `odlcuda`_ is available for Python 3.5 in the "odlgroup" conda channel.

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

**Build from source:**

Clone the ``odlcuda`` GitHub repository:

.. code-block:: bash

    $ git clone https://github.com/odlgroup/odlcuda.git

After that, follow the `build instructions there <https://github.com/odlgroup/odlcuda.git>`_.


ASTRA for X-ray tomography
--------------------------
To calculate fast forward and backward projections for image reconstruction in X-ray tomography, install the `ASTRA tomography toolbox`_.
ASTRA projectors are fully supported in ODL.

You can try using the conda package, but we can give no guarantee that it works out of the box:

.. code-block:: bash

    $ conda install -c astra-toolbox astra-toolbox


STIR for emission tomography
----------------------------
For applications in emission tomography, i.e. PET or SPECT, install STIR_.
Support for STIR is currently very limited.


Issues
======
If you have any problems during installation, consult the help in the :ref:`FAQ <FAQ>`.
If that does not help, `make an issue on GitHub`_ or send us an email (odl@math.kth.se) and we'll try to assist you promptly.


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
.. _odlcuda: https://github.com/odlgroup/odlcuda
.. _CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
.. _ASTRA tomography toolbox: https://github.com/astra-toolbox/astra-toolbox
.. _STIR: https://github.com/UCL/STIR
