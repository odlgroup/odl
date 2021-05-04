.. _development-workflow:

####################
Development workflow
####################

You already have your own forked copy of the `ODL`_ repository by following :ref:`forking`. You have
:ref:`set-up-fork`. You have configured Git according to :ref:`configure-git`.  Now you are ready
for some real work.

Workflow summary
================

In what follows we'll refer to the upstream ODL ``master`` branch, as
"trunk".

* Don't use your ``master`` branch for anything. Consider deleting it.
* When you are starting a new set of changes, fetch any changes from trunk,
  and start a new *feature branch* from that.
* Make a new branch for each separable set of changes |emdash| "one task, one
  branch" (see `IPython Git workflow`_).
* Name your branch for the purpose of the changes - e.g.
  ``issue-128__performance_tests`` or ``refactor_array_tests``.
  Use the ``issue-<number>__`` prefix for existing issues.
* If you are fixing a bug or implement a new feature, consider creating an issue on
  the `ODL issue tracker`_ first.
* *Never* merge trunk or any other branches into your feature branch while you are working.
* If you do find yourself merging from trunk, :ref:`rebase-on-trunk` instead.
* Ask on the `ODL mailing list`_ if you get stuck.
* Ask for code review!

This way of working helps to keep the project well organized, with readable history.
This in turn makes it easier for project maintainers (that might be you) to see
what you've done, and why you did it.

See `Linux Git workflow`_ and `IPython Git workflow`_ for some explanation.

Consider deleting your master branch
====================================

It may sound strange, but deleting your own ``master`` branch can help reduce
confusion about which branch you are on.  See `deleting master on GitHub`_ for
details.

.. _update-mirror-trunk:

Update the mirror of trunk
==========================

First make sure that :ref:`linking-to-upstream` is done.

From time to time you should fetch the upstream (trunk) changes from GitHub:

.. code-block:: bash

   $ git fetch upstream

This will pull down any commits you don't have, and set the remote branches to
point to the right commit.  For example, "trunk" is the branch referred to by
(remote/branchname) ``upstream/master`` - and if there have been commits since
you last checked, ``upstream/master`` will change after you do the fetch.

.. _make-feature-branch:

Make a new feature branch
=========================

When you are ready to make some changes to the code, you should start a new
branch.  Branches that are for a collection of related edits are often called
"feature branches".

Making an new branch for each set of related changes will make it easier for
someone reviewing your branch to see what you are doing.

Choose an informative name for the branch to remind yourself and the rest of us
what the changes in the branch are for, for example ``add-ability-to-fly`` or
``issue-42__fix_all_bugs``.

Is your feature branch mirroring an issue on the `ODL issue tracker`_? Then prepend your branch
name with the prefix ``issue-<number>__``, where ``<number>`` is the ticket number of the issue
you are going to work on.
If there is no existing issue that corresponds to the code you're about to write, consider
creating a new one. In case you are fixing a bug or implementing a feature, it is best to get in
contact with the maintainers as early as possible. Of course, if you are only playing around, you
don't need to create an issue and can name your branch however you like.

.. code-block:: bash

     # Update the mirror of trunk
   $ git fetch upstream
     # Make new feature branch starting at current trunk
   $ git branch my-new-feature upstream/master
   $ git checkout my-new-feature

Generally, you will want to keep your feature branches on your public GitHub
fork of ODL. To do this, you `git push`_ this new branch up to your GitHub repo.
Generally (if you followed the instructions in these pages, and by default), Git will have a link
to your GitHub repo, called ``origin``. You push up to your own repo on GitHub with

.. code-block:: bash

   $ git push origin my-new-feature

In git >= 1.7 you can ensure that the link is correctly set by using the
``--set-upstream`` option:

.. code-block:: bash

   $ git push --set-upstream origin my-new-feature

From now on Git will know that ``my-new-feature`` is related to the ``my-new-feature`` branch in
the GitHub repo.

.. _edit-flow:

The editing workflow
====================

Overview
--------

.. code-block:: bash

     # hack hack
   $ git add my_new_file
   $ git commit -m "BUG: fix all bugs"
   $ git push

In more detail
--------------

#. Make some changes.
#. See which files have changed with ``git status`` (see `git status`_).
   You'll see a listing like this one::

    On branch my-new-feature
    Changed but not updated:
      (use "git add <file>..." to update what will be committed)
      (use "git checkout -- <file>..." to discard changes in working directory)

            modified:   README

    Untracked files:
      (use "git add <file>..." to include in what will be committed)

            INSTALL

    no changes added to commit (use "git add" and/or "git commit -a")


#. Check what the actual changes are with ``git diff`` (see `git diff`_).
#. Add any new files to version control ``git add new_file_name`` (see `git add`_).
#. To commit all modified files into the local copy of your repo, do
   ``git commit -am "A commit message"``. Note the ``-am`` options to ``commit``. The ``m`` flag
   just signals that you're going to type :ref:`commit_message` on the command line.  The ``a``
   flag |emdash| you can just take on faith |emdash| or see `why the -a flag?`_ |emdash| and the
   helpful use-case description in the `tangled working copy problem`_. The `git commit`_ manual
   page might also be useful.
#. To push the changes up to your forked repo on GitHub, perform a ``git push`` (see `git push`_).

.. _commit_message:

The commit message
------------------
Bear in mind that the commit message will be part of the history of the repository,
shown by typing ``git log``, so good messages will make the history readable and searchable.
Don't see the commit message as an annoyance, but rather as an important part of
your contribution.

We appreciate if you follow the following style:

#. Start your commit with an `acronym`_, e.g., ``BUG``, ``TST`` or ``STY`` to
   indicate what kind of modification you make.
#. Write a one-line summary of your modification no longer than 50 characters.
   If you have a hard time summarizing you changes, maybe you need to split up
   the commit into parts.

   Use imperative style, i.e. write ``add super feature`` or ``fix horrific bug``
   rather than ``added, fixed ...``. This saves two characters for something else.

   Don't use markdown. You can refer to issues by writing ``#12``. You can even have GitHub
   automatically close an issue by writing ``closes #12``. This happens once your commit has
   made its way into ``master`` (usually after merging the pull request).
#. (optional) Write an extended summary. Describe why these changes are
   necessary and what the new code does better than the old one.

Ask for your changes to be reviewed or merged
=============================================

When you are ready to ask for someone to review your code and consider a merge:

#. Go to the URL of your forked repo, say
   ``http://github.com/your-user-name/odl``.
#. Use the "Switch branches/tags" dropdown menu near the top left of the page to
   select the branch with your changes:

   .. image:: branch-dropdown.png


#. Click on the "New Pull Request" button:

   .. image:: new-pull-request-button.png


   Enter a title for the set of changes, and some explanation of what you've
   done.  Say if there is anything you'd like particular attention for - like a
   complicated change or some code you are not happy with.

   If you don't think your request is ready to be merged, just say so in your
   pull request message.  This is still a good way of getting some preliminary
   code review.

   See also: https://help.github.com/articles/using-pull-requests/

Some other things you might want to do
======================================

Delete a branch on GitHub
-------------------------

.. code-block:: bash

   $ git checkout master
     # delete branch locally
   $ git branch -D my-unwanted-branch
     # delete the remote branch on GitHub
   $ git push origin :my-unwanted-branch

Note the colon ``:`` before ``test-branch``.

See also: http://github.com/guides/remove-a-remote-branch

Several people sharing a single repository
------------------------------------------

If you want to work on some stuff with other people, where you are all
committing into the same repository, or even the same branch, then just
share it via GitHub.

First fork ODL into your account, as from :ref:`forking`.

Then, go to your forked repository GitHub page, say ``http://github.com/your-user-name/odl``.

Click on "Settings" -> "Collaborators" button, and invite other people the repo as a collaborator.
Once they have accepted the invitation, they can do

.. code-block:: bash

    $ git clone git@githhub.com:your-user-name/odl.git

Remember that links starting with ``git@`` use the ssh protocol and are read-write; links starting
with ``https://`` are read-only.

Your collaborators can then commit directly into that repo with the usual

.. code-block:: bash

     $ git commit -am "ENH: improve code a lot"
     $ git push origin master  # pushes directly into your repo

See also: https://help.github.com/articles/inviting-collaborators-to-a-personal-repository/

Explore your repository
-----------------------

To see a graphical representation of the repository branches and commits, use a `Git GUI`_ like
``gitk`` shipped with Git or ``QGit`` included in KDE:

.. code-block:: bash

   $ gitk --all

To see a linear list of commits for this branch, invoke

.. code-block:: bash

   $ git log

You can also look at the `Network graph visualizer`_ for your GitHub repo.

Finally the :ref:`fancy-log` ``fancylog`` alias will give you a reasonable text-based graph of the
repository.

.. _rebase-on-trunk:

Rebase on trunk
---------------

Let's say you thought of some work you'd like to do. You :ref:`update-mirror-trunk` and
:ref:`make-feature-branch` called ``cool-feature``. At this stage trunk is at some commit, let's
call it E. Now you make some new commits on your ``cool-feature`` branch, let's call them A, B,
C. Maybe your changes take a while, or you come back to them after a while. In the meantime, trunk
has progressed from commit E to commit (say) G::

          A---B---C cool-feature
         /
    D---E---F---G trunk

Now you consider merging trunk into your feature branch, and you remember that this page sternly
advises you not to do that, because the history will get messy. Most of the time you can just ask
for a review, and not worry that trunk has got a little ahead. But sometimes, the changes in trunk
might affect your changes, and you need to harmonize them. In this situation, you may prefer to do
a rebase.

Rebase takes your changes (A, B, C) and replays them as if they had been made to the current state
of ``trunk``. In other words, in this case, it takes the changes represented by A, B, C and replays
them on top of G. After the rebase, your history will look like this::

                  A'--B'--C' cool-feature
                 /
    D---E---F---G trunk

See `rebase without tears`_ for more detail.

To do a rebase on trunk:

.. code-block:: bash

     # Update the mirror of trunk
   $ git fetch upstream

     # go to the feature branch
   $ git checkout cool-feature

     # make a backup in case you mess up
   $ git branch tmp cool-feature

     # rebase cool-feature onto trunk
   git rebase --onto upstream/master upstream/master cool-feature

In this situation, where you are already on branch ``cool-feature``, the last
command can be written more succinctly as

.. code-block:: bash

   $ git rebase upstream/master

When all looks good you can delete your backup branch:

.. code-block:: bash

   $ git branch -D tmp

If it doesn't look good you may need to have a look at :ref:`recovering-from-mess-up`.

If you have made changes to files that have also changed in trunk, this may generate merge conflicts
that you need to resolve - see the `git rebase`_ manual page for some instructions at the end of the
"Description" section. There is some related help on merging in the Git user manual - see
`resolving a merge`_.

.. _recovering-from-mess-up:

Recovering from mess-ups
------------------------

Sometimes, you mess up merges or rebases. Luckily, in Git it is relatively straightforward to recover
from such mistakes.

If you mess up during a rebase:

.. code-block:: bash

   $ git rebase --abort

If you notice you messed up after the rebase:

.. code-block:: bash

     # reset branch back to the saved point
   $ git reset --hard tmp

If you forgot to make a backup branch:

.. code-block:: bash

     # look at the reflog of the branch
   $ git reflog show cool-feature

   8630830 cool-feature@{0}: commit: BUG: io: close file handles immediately
   278dd2a cool-feature@{1}: rebase finished: refs/heads/my-feature-branch onto 11ee694744f2552d
   26aa21a cool-feature@{2}: commit: BUG: lib: make seek_gzip_factory not leak gzip obj
   ...


     # reset the branch to where it was before the botched rebase
   $ git reset --hard cool-feature@{2}

.. _rewriting-commit-history:

Rewriting commit history
------------------------

.. note::

   Do this only for your own feature branches.

There's an embarrassing typo in a commit you made? Or perhaps the you made several false starts you
would like the posterity not to see.

This can be fixed via *interactive rebasing*.

Suppose that the commit history looks like this:

.. code-block:: bash

    $ git log --oneline
    eadc391 Fix some remaining bugs
    a815645 Modify it so that it works
    2dec1ac Fix a few bugs + disable
    13d7934 First implementation
    6ad92e5 * masked is now an instance of a new object, MaskedConstant
    29001ed Add pre-nep for a copule of structured_array_extensions.
    ...

and ``6ad92e5`` is the last commit in the ``cool-feature`` branch. Suppose we
want to make the following changes:

* Rewrite the commit message for ``13d7934`` to something more sensible.
* Combine the commits ``2dec1ac``, ``a815645``, ``eadc391`` into a single one.

We do as follows:

.. code-block:: bash

      # make a backup of the current state
    $ git branch tmp HEAD
      # interactive rebase
    $ git rebase -i 6ad92e5

This will open an editor with the following text in it::

    pick 13d7934 First implementation
    pick 2dec1ac Fix a few bugs + disable
    pick a815645 Modify it so that it works
    pick eadc391 Fix some remaining bugs

    # Rebase 6ad92e5..eadc391 onto 6ad92e5
    #
    # Commands:
    #  p, pick = use commit
    #  r, reword = use commit, but edit the commit message
    #  e, edit = use commit, but stop for amending
    #  s, squash = use commit, but meld into previous commit
    #  f, fixup = like "squash", but discard this commit's log message
    #
    # If you remove a line here THAT COMMIT WILL BE LOST.
    # However, if you remove everything, the rebase will be aborted.
    #

To achieve what we want, we will make the following changes to it::

    r 13d7934 First implementation
    pick 2dec1ac Fix a few bugs + disable
    f a815645 Modify it so that it works
    f eadc391 Fix some remaining bugs

This means that (i) we want to edit the commit message for ``13d7934``, and (ii) collapse the last
three commits into one. Now we save and quit the editor.

Git will then immediately bring up an editor for editing the commit
message. After revising it, we get the output::

    [detached HEAD 721fc64] FOO: First implementation
     2 files changed, 199 insertions(+), 66 deletions(-)
    [detached HEAD 0f22701] Fix a few bugs + disable
     1 files changed, 79 insertions(+), 61 deletions(-)
    Successfully rebased and updated refs/heads/my-feature-branch.

and the history looks now like this::

     0f22701 Fix a few bugs + disable
     721fc64 ENH: Sophisticated feature
     6ad92e5 * masked is now an instance of a new object, MaskedConstant

If it went wrong, recovery is again possible as explained :ref:`above
<recovering-from-mess-up>`.

.. include:: links.inc
