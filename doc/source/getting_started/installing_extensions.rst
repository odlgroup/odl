.. _installing_odl_extensions:

Compiled extensions
===================
There are several compiled extensions to ODL.
Some of them can be installed using ``conda`` or ``pip``, others require manual compilation.
This section assumes that you have a working installation of python and ODL.


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
