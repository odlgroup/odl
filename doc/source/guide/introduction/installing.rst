##########
Installing
##########

Installing odl is intended to be straightforward, and this guide is meant for new users. For a minimal installation you should do the following steps:

1. `Install Python`_
2. `Install ODL`_
3. `Run tests`_

If you have done any step before, you can skip it, of course.

Install Python
==============
To begin with, you need a Python distribution. This is included by default on most Linux distributions, but may need to be installed.

Anaconda
--------
If you are a Python novice using Windows, we recommend that you install a full package such as Anaconda. It can be installed from `Anaconda's webpage <https://www.continuum.io/downloads>`_. Once installed, run in a console

.. code-block:: bash

   user$ conda update --all
		
to make sure you have the latest versions of all packages

What Python version to use
--------------------------
Any modern Python distribution should do for the core library, but some extensions require CPython (the standard Python distribution).

ODL fully supports both Python 2 and Python 3, so if you already have a Python distribution (as is the case on most linux distributions) it may be a good idea to keep using it. If you are making a new installation we recommend using Python 3, since Python 2 support will be discontinued in 2020.

Development enviroment
----------------------
Since ODL is object oriented, using an Integrated Development Environment (IDE) is recommended, but not required. The most popular ones are `Spyder
<https://pythonhosted.org/spyder/>`_ which works on all major platforms and can be installed through ``pip``, and `PyCharm
<https://www.jetbrains.com/pycharm/>`_ which can be integrated with any text editor of your choice, such as EMACS or Vim.

Install ODL
===========
There are two ways to install ODL, either using ``pip`` or from source. Unless you plan to contribute to the library we recommend using ``pip`` directly.

Requirements
------------
You need to `install pip
<https://pip.pypa.io/en/stable/installing/#installation>`_ to be able to install ODL and its dependencies from the `Python package index
<https://pypi.python.org/pypi>`_. Usually, ``pip`` is bundled with Python, so there is no action required. Check if simply running ``pip`` shows a help message or an error.

**General remark:** Try to install as many dependencies as possible through your distribution package manager (on Linux) or with a complete scientific Python distribution like Anaconda. Packages installed through ``pip`` can cause conflicts with identical packages installed by a different mechanism (our basic dependencies are largely unproblematic, though).

**Minimal dependencies**

- numpy_ >= 1.9
- scipy_ >= 0.14
- python-future_ >= 0.14

These packages are automatically installed through ``pip`` alongside ODL if they are not found on the system or the version requirement is not met.

.. _numpy: https://github.com/numpy/numpy
.. _scipy: https://github.com/scipy/scipy
.. _python-future: https://pypi.python.org/pypi/future/

Installing ODL through pip
--------------------------
For basic installation without extra dependencies, run

.. code-block:: bash

   user$ pip install odl

If you don't have administrator rights on your machine, you can install packages locally as a user:

.. code-block:: bash

   user$ pip install --user odl

Extra dependencies
------------------
Additional dependencies can be specified in square brackets, separated by commas (no spaces!):

.. code-block:: bash

   user$ pip install odl[dep1,dep2]

Possible choices:

- ``all`` : Install with all extra dependencies.
- ``show`` : Install matplotlib_ to enable displaying capabilities.
- ``fft`` : Install pyFFTW_ for fast Fourier transforms. Note that this requires the FFTW_ library to be available on your system.
  Note also that even without this dependency, FFTs can be computed with Numpy's FFT library.
- ``wavelets`` : Install PyWavelets_ for wavelet transforms.

.. _matplotlib: http://matplotlib.org/
.. _pyFFTW: https://pypi.python.org/pypi/pyFFTW
.. _FFTW: http://fftw.org/
.. _PyWavelets: https://pypi.python.org/pypi/PyWavelets

Installing ODL from source
--------------------------
This installation method is intended for developers who want to make changes to the code. It assumes that the Git_ version control system is available on your system. If you do not have Git installed, see `Install Git`_. You still need pip installed to perform the installation.

To get ODL, clone the repository with the command

.. code-block:: bash

   user$ git clone https://github.com/odlgroup/odl

No GitHub account is required for this step. For installation in a local user folder, enter the top-level directory of the cloned repository and run

.. code-block:: bash

   user$ pip install --user --editable .

**Don't forget the "." (dot) at the end** - it refers to the current directory, the location from where ``pip`` is supposed to install ODL.

We recommend the ``--editable`` option (can be shortened to ``-e``) since it installs a link instead of copying the files to your Python packages location. This way local changes to the code (e.g. after a ``git pull``) take immediate effect without reinstall.

If you prefer a system-wide installation, leave out the ``--user`` option. To do this, you need administrator rights.

.. _Git: http://www.git-scm.com/

Install Git
-----------
You can download git using the following commands/links.

================ =============
Debian / Ubuntu  ``sudo apt-get install git``
Fedora           ``sudo yum install git``
Windows          Download and install msysGit_
OS X             Use the git-osx-installer_
================ =============

.. _msysgit: http://code.google.com/p/msysgit/downloads/list
.. _git-osx-installer: http://code.google.com/p/git-osx-installer/downloads/list

**Helpful links**

Have a look at the github install help pages available from `github help`_

There are good instructions here: http://book.git-scm.com/2_installing_git.html

.. _github help : https://help.github.com/

Further Information
-------------------
See :ref:`Contributing to ODL <contributing>` for more information.


Run tests
=========
Unit tests in ODL are based on pytest_, and coverage reports are created by the coverage_ module. These packages are installed if you run

.. code-block:: bash

   user$ pip install --user -e .[testing]

Now you can check that everything was installed properly by running

.. code-block:: bash

   user$ py.test

.. _pytest: https://pypi.python.org/pypi/pytest
.. _coverage: https://pypi.python.org/pypi/coverage/

Compiled extensions
===================
There are several extensions to ODL that require you to compile external code, these include

CUDA backend for linear arrays
------------------------------
If you also wish to use the (optional) CUDA extensions you need to run

.. code-block:: bash

    user$ git submodule update --init --recursive
    user$ cd odlpp

From here follow the instructions in odlpp_ to build and install it.

.. _odlpp: https://github.com/odlgroup/odlpp

ASTRA for X-ray tomography
--------------------------

To calculate forward and backward projections for image reconstruction in X-ray tomography, install the
`ASTRA tomography toolbox`_. ASTRA projectors are fully supported
in ODL.

.. _ASTRA tomography toolbox: https://github.com/astra-toolbox/astra-toolbox

STIR for emission tomography
----------------------------
For applications in emission tomography, i.e. PET or SPECT, install STIR_. Support
for STIR is currently very limited.

.. _STIR: https://github.com/UCL/STIR

Issues
======
If you have any problems during installation, consult the help in the :ref:`FAQ <FAQ>`. If that does not help, make an issue on GitHub and we'll try to assist you promptly.
