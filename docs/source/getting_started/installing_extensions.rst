.. _installing_odl_extensions:

#########################
Installing ODL extensions
#########################


.. _installing_odl_extensions__compiled:

Compiled extensions
===================
There are several compiled extensions to ODL.
Some of them can be installed using ``conda`` or `pip`_, others require manual compilation.
This section assumes that you have a working installation of python and ODL.


.. _installing_odl_extensions__astra:

ASTRA for X-ray tomography
==========================
To calculate fast forward and backward projections for image reconstruction in X-ray tomography, install the `ASTRA tomography toolbox <https://github.com/astra-toolbox/astra-toolbox>`_.
ASTRA projectors are fully supported in ODL.

Astra is most easily installed using conda:

.. code-block:: bash

    $ conda install -c astra-toolbox astra-toolbox

For further instructions, check `the ASTRA GitHub page <https://github.com/astra-toolbox/astra-toolbox>`_.



CUDA backend for linear arrays
==============================

.. warning::
    This plugin is dysfunctional with ODL master since the API change introduced by :pull:`1088`.
    It can be used with older versions of ODL (e.g., with the current release).
    The plugin will be replaced by CuPy in short (:pull:`1231`).

The `odlcuda`_ backend for fast array calculations on CUDA requires the `CUDA toolkit`_ (on Linux: use your distro package manager) and a CUDA capable graphics card with compute capability of at least 3.0.
Search `this table <https://en.wikipedia.org/wiki/CUDA#GPUs_supported>`_ for your model.

Building from source
--------------------
You have two options of building ``odlcuda`` from source.
For both, first clone the ``odlcuda`` GitHub repository and enter the new directory:

.. code-block:: bash

    $ git clone https://github.com/odlgroup/odlcuda.git
    $ cd odlcuda

1. **Using conda build**

   This is the simpler option and should work on any Linux or MacOS system (we currently have no Windows build recipe, sorry).

   To build the conda recipe, you should be **in the root conda environment** (see :ref:`installing_odl_conda__installing_anaconda` for details) and in the top-level directory of your ``odlcuda`` clone.
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

       $ conda install --use-local odlcuda


2. **Manually with CMake**

   This option requires more manual work but is known to work on all platforms.

   See `here <https://github.com/odlgroup/odlcuda.git>`_ for build instructions.
   You may want to use include and library paths (GCC, boost, ...) of a conda enviroment and install the package in it.

A simple test if this build of ``odlcuda`` works, you can run

.. code-block:: bash

    $ python -c "import odl; odl.rn(3, impl='cuda').element()"

If you get a ``KeyError: 'cuda'``, then something went wrong with the package installation since it cannot be imported.
If the above command instead raises a ``MemoryError`` or similar, your graphics card is not properly configured, and you should solve that issue first.


.. _pip: https://pip.pypa.io/en/stable/

.. _odlcuda: https://github.com/odlgroup/odlcuda
.. _CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
.. _ASTRA: https://github.com/astra-toolbox/astra-toolbox
