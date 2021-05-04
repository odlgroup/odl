.. _configure-git:

===============
 Configure Git
===============

.. _git-config-basic:

Overview
========

Your personal Git configurations are saved in the ``.gitconfig`` file in
your home directory.

Here is an example ``.gitconfig`` file::

  [user]
          name = Your Name
          email = you@yourdomain.example.com

  [alias]
          ci = commit -a
          co = checkout
          st = status
          stat = status
          br = branch
          wdiff = diff --color-words
          show-conflicts = !git --no-pager diff --name-only --diff-filter=U

  [core]
          editor = nano

  [merge]
          summary = true

You can edit this file directly or you can use the ``git config --global``
command:

.. code-block:: bash

   $ git config --global user.name "Your Name"
   $ git config --global user.email you@yourdomain.example.com
   $ git config --global alias.ci "commit -a"
   $ git config --global alias.co checkout
   $ git config --global alias.st "status"
   $ git config --global alias.stat "status"
   $ git config --global alias.br branch
   $ git config --global alias.wdiff "diff --color-words"
   $ git config --global alias.show-conflicts "!git --no-pager diff --name-only --diff-filter=U"
   $ git config --global core.editor nano
   $ git config --global merge.summary true

To set up on another computer, you can copy your ``~/.gitconfig`` file,
or run the commands above.

In detail
=========

user.name and user.email
------------------------

It is good practice to tell Git_ who you are, for labeling any changes
you make to the code. The simplest way to do this is from the command
line:

.. code-block:: bash

   $ git config --global user.name "Your Name"
   $ git config --global user.email you@yourdomain.example.com

This will write the settings into your Git configuration file,  which
should now contain a user section with your name and email::

  [user]
        name = Your Name
        email = you@yourdomain.example.com

Of course you'll need to replace ``Your Name`` and ``you@yourdomain.example.com``
with your actual name and email address.

Aliases
-------

You might well benefit from some aliases to common commands.

For example, you might well want to be able to shorten ``git checkout``
to ``git co``.  Or you may want to alias ``git diff --color-words``
(which gives a nicely formatted output of the diff) to ``git wdiff``.
Another useful alias is ``git show-conflicts`` which displays files that
are currently in conflict.

The ``git config --global`` commands

.. code-block:: bash

   $ git config --global alias.ci "commit -a"
   $ git config --global alias.co checkout
   $ git config --global alias.st "status -a"
   $ git config --global alias.stat "status -a"
   $ git config --global alias.br branch
   $ git config --global alias.wdiff "diff --color-words"
   $ git config --global alias.show-conflicts "!git --no-pager diff --name-only --diff-filter=U"

will create an ``alias`` section in your ``.gitconfig`` file with contents
like this::

  [alias]
          ci = commit -a
          co = checkout
          st = status -a
          stat = status -a
          br = branch
          wdiff = diff --color-words
          show-conflicts = !git --no-pager diff --name-only --diff-filter=U

Editor
------

You may also want to make sure that your favorite editor is used:

.. code-block:: bash

   $ git config --global core.editor nano

Merging
-------

To enforce summaries when doing merges (``~/.gitconfig`` file again)::

   [merge]
      log = true

Or from the command line:

.. code-block:: bash

   $ git config --global merge.log true

.. _fancy-log:

Fancy log output
----------------

This is a very nice alias to get a fancy log output; it should go in the
``alias`` section of your ``.gitconfig`` file::

    fancylog = log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)[%an]%Creset' --abbrev-commit --date=relative

You use the alias with:

.. code-block:: bash

   $ git fancylog

and it gives graph / text output something like this (but with color!)::

    * a105abc - (HEAD -> master, upstream/master) Revert "MAINT: replace deprecated pngmath extension by imgmath" (50 minutes ago) [Holger Kohr]
    * 05168c9 - MAINT: replace deprecated pngmath extension by imgmath (53 minutes ago) [Holger Kohr]
    * f654c3d - DOC: update README and description in setup.py a bit (19 hours ago) [Holger Kohr]
    *   d097c7b - Merge pull request #436 from odlgroup/issue-435__parallel2d_rotation (19 hours ago) [Holger Kohr]
    |\
    | * 180ba96 - (upstream/issue-435__parallel2d_rotation, issue-435__parallel2d_rotation) TST: Add test for angle conventions of projectors (24 hours ago) [Jonas Adler]
    | * de2ab55 - BUG: fix behaviour of show with nonuniform data (26 hours ago) [Jonas Adler]
    | * a979666 - BUG: fix rotation by 90 degrees for 2d parallel (27 hours ago) [Holger Kohr]
    |/
    *   ecfd306 - Merge pull request #444 from odlgroup/issue-443__uniform_partition (29 hours ago) [Holger Kohr]
    |\
    | * 024552f - MAINT: replace 10 ** -10 with 1e-10 in domain_test.py (29 hours ago) [Holger Kohr]
    | * 032b89d - ENH: allow single tuple for nodes_on_bdry in uniform_sampling for 1d (29 hours ago) [Holger Kohr]
    | * 85dda52 - ENH: add atol to IntervalProd.contains_all (29 hours ago) [Holger Kohr]
    | * bdaef8c - ENH: make uniform_partition more flexible (29 hours ago) [Holger Kohr]
    | * 72b4bd5 - MAINT: use odl.foo instead of from odl import foo in partition_test.py (2 days ago) [Holger Kohr]
    | * 11ec155 - MAINT: fix typo in grid.py (2 days ago) [Holger Kohr]
    | * dabc917 - MAINT: change tol parameter in IntervalProd to atol (2 days ago) [Holger Kohr]
    * |   e59662c - Merge pull request #439 from odlgroup/issue-409__element_noop (29 hours ago) [Jonas Adler]
    |\ \
    | |/
    |/|
    | * 1d41554 - API: enforce element(vec) noop (8 days ago) [Jonas Adler]
    * |   34d4e74 - Merge pull request #438 from odlgroup/issue-437__discr_element_broadcast (8 days ago) [Jonas Adler]
    |\ \
    | |/
    |/|
    | * e09bfa9 - ENH: allow broadcasting in discr element (8 days ago) [Jonas Adler]

Thanks to Yury V. Zaytsev for posting it.

.. include:: links.inc
