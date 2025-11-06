.. _set-up-fork:

==================
 Set up your fork
==================

First follow the instructions for :ref:`forking`.

Overview
========

.. code-block:: bash

   $ git clone git@github.com:your-user-name/odl.git
   $ cd odl
   $ git remote add upstream https://github.com/odlgroup/odl.git

In detail
=========

Clone your fork
---------------

#. Clone your fork to the local computer with

   .. code-block:: bash

      $ git clone git@github.com:your-user-name/odl.git

#. Investigate.  Change directory to your new repo: ``cd odl``. Then
   ``git branch -a`` to show you all branches.  You'll get something
   like this::

      * master
      remotes/origin/master

   This tells you that you are currently on the ``master`` branch, and
   that you also have a ``remote`` connection to ``origin/master``.
   What remote repository is ``remote/origin``? Try ``git remote -v`` to
   see the URLs for the remote.  They will point to your GitHub fork.

   Now you want to connect to the upstream `ODL GitHub`_ repository, so
   you can merge in changes from trunk.

.. _linking-to-upstream:

Linking your repository to the upstream repo
--------------------------------------------

.. code-block:: bash

   $ cd odl
   $ git remote add upstream https://github.com/odlgroup/odl.git

``upstream`` here is just the arbitrary name we're using to refer to the
main ODL repository at `ODL GitHub`_.

Note that we've used ``https://`` for the URL rather than ``git@``.  The ``https://`` URL is
read-only.  This means we that we can't accidentally (or deliberately) write to the upstream repo,
and we are only going to use it to merge into our own code.

Just for your own satisfaction, show yourself that you now have a new
"remote", with ``git remote -v show``, giving you something like::

   upstream	https://github.com/odlgroup/odl.git (fetch)
   upstream	https://github.com/odlgroup/odl.git (push)
   origin	git@github.com:your-user-name/odl.git (fetch)
   origin	git@github.com:your-user-name/odl.git (push)

.. include:: links.inc

