.. _installing_odl:

##############
Installing ODL
##############

This guide will go through all steps necessary for a full ODL installation, starting from nothing more than a working operating system (Linux, MacOS or Windows).


.. _installing_odl__tldr:

TL;DR
=====
If you already have a working python environment, ODL and some basic dependencies can be installed using either `pip`_:

.. code-block:: bash

    $ pip install odl[testing,show]

or conda:

.. code-block:: bash

    $ conda install -c odlgroup odl matplotlib pytest scikit-image spyder

After installation, the installation can be verified by running the tests:

.. code-block:: bash

    $ python -c "import odl; odl.test()"


.. _installing_odl__introduction:

Introduction
============

Installing ODL is intended to be straightforward, and this guide is meant for new users.
For a working installation you should perform the following steps:

1. Install a Python interpreter
2. Install ODL and its dependencies
3. (optional) Install extensions for more functionality
4. (optional) Run the tests


.. _installing_odl__consider_anaconda:

Consider using Anaconda
=======================
We currently recommend to use `Anaconda`_ on all platforms since it offers the best out-of-the-box installation and run-time experience.
Anaconda also has other benefits, for example the possibility to work in completely isolated Python environments with own installed packages, thereby avoiding conflicts with system-wide installed packages.
Furthermore, Anaconda cooperates with ``pip`` (see below), i.e. packages can be installed with both Anaconda's internal mechanism and ``pip`` without conflicts.

Alternatively, packages can be installed with `pip`_ in a user's location, which should also avoid conflicts.
We will provide instructions for this alternative.

Another possibility is to use `virtualenv`_, which can be seen as a predecessor to Anaconda.
Following the ``pip`` installation instructions in a ``virtualenv`` without the ``--user`` option works very well in our experience, but we do not provide explicit instructions for this variant.


.. _installing_odl__python_version:

Which Python version to use?
============================
Any modern Python distribution supporting `NumPy`_ and `SciPy`_ should work for the core library, but some extensions require CPython (the standard Python distribution).

ODL fully supports both Python 2 and Python 3.
If you choose to use your system Python interpreter (the "pip install as user" variant), it may be a good idea to stick with the default one, i.e. the one invoked by the ``python`` command on the command line.
Otherwise, we recommend using Python 3, since Python 2 support will be discontinued in 2020.


.. _installing_odl__development_environment:

Development environment
=======================
Since ODL is object-oriented, using an Integrated Development Environment (IDE) is recommended, but not required.
The most popular ones are `Spyder`_ which works on all major platforms and can be installed through both ``conda`` and ``pip``, and `PyCharm`_ which can be integrated with any text editor of your choice, such as Emacs or Vim.


.. _installing_odl__in_depth_guides:

In-depth guides
===============
If you are a new user or need more a detailed installation guide, we provide support for the following installation methods:

1. :ref:`installing_odl_conda` (recommended for users)
2. :ref:`installing_odl_pip`
3. :ref:`installing_odl_source` (recommended for developers)

To further extend ODL capability, a :ref:`large set of extensions<installing_odl_extensions>` can also be installed.


.. _installing_odl__issues:

Issues
======
If you have any problems during installation, consult the help in the :ref:`FAQ <FAQ>`.
If that does not help, `make an issue on GitHub <https://github.com/odlgroup/odl/issues>`_ or send us an email (odl@math.kth.se) and we'll try to assist you promptly.


.. _Anaconda: https://anaconda.org/

.. _virtualenv: https://virtualenv.pypa.io/en/stable/
.. _pip: https://pip.pypa.io/en/stable/

.. _Spyder: https://github.com/spyder-ide/spyder
.. _PyCharm: https://www.jetbrains.com/pycharm/

.. _NumPy: http://www.numpy.org/
.. _SciPy: https://www.scipy.org/
