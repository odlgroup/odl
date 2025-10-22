.. _installing_odl_source:

==========================
Installing ODL from source
==========================
This installation method is intended for developers who want to make changes to the code and users that need the cutting edge.

TL;DR
=====
Instructions for the impatient:

- Clone ODL from git:

  .. code-block:: bash

     $ git clone https://github.com/odlgroup/odl

- Install ODL

  .. code-block:: bash

    $ cd odl
    $ pip install [--user] --editable .

  Don't use the ``--user`` option together with ``conda``.

- Install the :ref:`extensions you want <installing_odl_extensions>`.


Introduction
============
This guide assumes that the `Git`_ version control system is available on your system; for up-to-date instructions, consult the `Git installation instructions <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.
You also need `pip`_ to perform the installation.

.. note::
    You should consider performing all described steps in a `conda environment <http://conda.pydata.org/docs/using/envs.html>`_ -- it gives you the same encapsulation benefits as developer that you would enjoy also as a user (no conflicting packages, free to choose Python version, ...).
    See the :ref:`installing_odl_conda__installing_anaconda` section for setup instructions.

To get ODL, navigate to a folder where you want the ODL repository to be stored and clone the repository with the command

.. code-block:: bash

   $ git clone https://github.com/odlgroup/odl

No GitHub account is required for this step.


In a conda environment
======================
This part assumes that you have activated a conda environment before (see :ref:`installing_odl_conda__installing_anaconda`).

You can choose to install dependencies first:

* On Linux/MacOS:

    .. code-block:: bash

        $ conda install nomkl numpy scipy future matplotlib

* On Windows:

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


.. note::
    **Don't forget the "." (dot) at the end** - it refers to the current directory, the location from where ``pip`` is supposed to install ODL.

.. note::
    We recommend the ``--editable`` option (can be shortened to ``-e``) since it installs a link instead of copying the files to your Python packages location.
    This way, local changes to the code (e.g. after a ``git pull``) take immediate effect after reloading the package, without requiring re-installation.


**Optional dependencies:**

You may also want to install optional dependencies:

.. code-block:: bash

    $ pip install --user .[testing, show]

Extra dependencies
------------------
As a developer, you may want to install further optional dependencies.
Consult the :ref:`pip <installing_odl_pip__extensions>` or :ref:`conda <installing_odl_conda__extensions>` guide for further instructions.

Running the tests
=================
Unit tests in ODL are based on `pytest`_.
They can be run either from within ``odl`` or by invoking ``pytest`` directly.

First, you need to install the testing dependencies using your favorite method below.

* Using conda:

    .. code-block:: bash

        $ conda install pytest

* Using pip:

    .. code-block:: bash

        $ pip install --user odl[testing]

Now you can check that everything was installed properly by running

.. code-block:: bash

   $ python -c "import odl; odl.test()"

.. note::
    If you have several versions of ODL and run this command in the top-level directory of an ODL clone, the tests in the repository will be run, not the ones in the installed package.

You can also use ``pytest`` directly in the root of your ODL clone:

.. code-block:: bash

   $ pytest

For more information on the tests, see :ref:`dev_testing`.

Further developer information
=============================
See :ref:`Contributing to ODL <contributing>` for more information.


.. _pip: https://pip.pypa.io/en/stable/
.. _Git: http://www.git-scm.com/
.. _pytest: https://pypi.python.org/pypi/pytest
