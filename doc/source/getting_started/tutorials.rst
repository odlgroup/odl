.. _tutorial:

#############
ODL tutorials
#############

Welcome to the ODL tutorials section.
Here, you will learn how to use ODL from the bottom up.
The first bunch of examples takes you through simple use cases and should make you familiar with the syntax and the "feel" of working with the framework.

.. note::
    This documentation section is work in progress.
    Tutorials will be added on a regular basis, and we try to keep them in sync with the code base.
    If you notice a bug in a notebook, please open an issue on the `issue tracker <https://github.com/odlgroup/odl/issues>`_.


How to use the tutorials
========================
Each tutorial is provided in two flavors - you can choose which one to use:

Option 1: `Jupyter Notebooks <https://jupyter.org/>`_ (recommended)
-------------------------------------------------------------------
A notebook is a structured file containing both static text (explanations, formulas, ...) and interactive content.
The big advantages of this format are that

- example code and accompanying text are kept together nicely,
- there is no need to switch to and from code and
- the code actually runs and can be changed at will.

This approach creates a very direct user experience and makes it easy to play around with or extend existing examples.

To install ``jupyter-notebook``, run

.. code-block:: bash

    conda install notebook

or

.. code-block:: bash

    pip install notebook

depending on your setup.

.. rubric:: Usage:

- Download a Jupyter notebook from a link in one of the tables below.
- Run ``jupyter-notebook /path/to/notebook-file.ipynb``.

This will open a browser tab with the notebook ready to be run. Click on "Help -> User Interface Tour" for an introduction into Jupyter - it's really easy!


Option 2: Python Code Files
---------------------------
A pure Python version of each notebook is available as alternative link in the each of the tables.
These files are auto-generated from the notebooks, so Markup and LaTeX formulas won't look as comprehensible as they would in the rendered notebook.

.. note::
    If you have a browser but not the ``jupyter-notebook`` package, you can view a static version of the notebook at `<https://nbviewer.jupyter.org/>`_.
    To do so, copy a notebook link and paste it into the text field at the notebook viewer front page.
    This will take you to a statically rendered version of the notebook.

.. rubric:: Usage:

- Download a Python script from a link in one of the tables below.
- Run the commands selectively in an interactive shell, for example in the IPython shell of `Spyder <https://github.com/spyder-ide/spyder>`_ by selecting parts and pressing ``F9``.
  Otherwise, if you run the script as a whole, a lot of useful output will not be displayed.


.. _tutorials_listing

=========

.. list-table::  **Basic ODL concepts**
    :name: table_basic_odl_concepts
    :header-rows: 1

    * - Jupyter notebook
      - Python script
      - Level

    * - :download:`ODL Basics.ipynb <notebooks/ODL Basics.ipynb>`
      - :download:`ODL Basics.py <code/ODL Basics.py>`
      - Beginner

    * - :download:`Exploring Vector Spaces.ipynb <notebooks/Exploring Vector Spaces.ipynb>`
      - :download:`Exploring Vector Spaces.py <code/Exploring Vector Spaces.py>`
      - Beginner

