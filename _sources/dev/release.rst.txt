.. _dev_release:

#######################
The ODL release process
#######################

This document is intended to give precise instructions on the process of making a release.
Its purpose is to avoid broken packages, broken documentation and many other things that can go wrong as a result of mistakes during the release process.
Since this is not everyday work and may be done under the stress of a (self-imposed) deadline, it is clearly beneficial to have a checklist to hold on to.

.. note::
    The instructions in this document are written from the perspective of Linux and may need adaption for other platforms.


.. _dev_rel_release_schedule:

1. Agree on a release schedule
------------------------------
This involves the "what" and "when" of the release process and fixes a feature set that is supposed to be included in the new version.
The steps are:

- Open an issue on the issue tracker using the title **Release X.Y.Z** (insert numbers, of course).
- Discuss and agree on a set of open PRs that should be merged and issues that should be resolved before making a release.
- Consider posting a shortened version of these instructions as a checklist on the issue page.
  It tends to be useful for keeping track of progress, and it is always satisfactory to tick off action points.

`This issue page <https://github.com/odlgroup/odl/issues/1335>`_ is a good template since it largely adheres to all points mentioned here.


.. _dev_rel_master_ok:

2. Make sure tests succeed and docs are built properly
------------------------------------------------------
When all required PRs are merged, ensure that the latest ``master`` branch is sane. Travis CI checks every PR, but certain things like CUDA cannot be tested there and must therefore undergo tests on a local machine, for at least Python 2.7 and one version of Python 3.

- Make a new test conda environment and install all dependencies:

  .. code-block:: bash

    conda create -n release36 python=3.6 nomkl numpy scipy future packaging pytest
    conda activate release36
    cd /path/to/odl_repo
    git fetch origin && git checkout origin/master
    pip install -e .

- Run the tests with ``pytest``, including doctests, examples documentation and large-scale tests:

  .. code-block:: bash

    pytest --examples --doctest-doc --largescale

- Run the tests again after installing ``pyfftw``, ``pywavelets`` and ``astra-toolbox``:

  .. code-block:: bash

     conda install pywavelets
     conda install -c conda-forge pyfftw
     pytest --largescale

- Run the alternative way of invoking the tests:

  .. code-block:: bash

     python -c "import odl; odl.test()"

- Repeat the steps for Python 2.7.
- Make sure the tests also run on the platforms you're currently *not* testing on.
  Ask a buddy maintainer if necessary.
- Build the documentation.
  This requires ``sphinx`` and the ``sphinxext`` submodule:

  .. code-block:: bash

    conda install sphinx sphinx_rtd_theme
    git submodule update --init --recursive
    cd doc && make clean
    cd source && python generate_doc.py
    cd ..
    make html 2>&1 |\
      grep -E "SEVERE|ERROR|WARNING" |\
      grep -E -v "more than one target found for|__eq__|document isn't included in any toctree"

  The last command builds the documentation and filters from the output all irrelevant warnings, letting through only the "proper" warnings and errors.
  If possible, *fix these remaining issues*.
- Glance the built documentation (usually in ``doc/_build``) for obvious errors.
- If there are test failures or documentation glitches, fix them and make a PR into the ``master`` branch.
  Do **not** continue with the next step until this step is finished!


.. _dev_rel_release_branch:

3. Make a release branch off of ``master``
------------------------------------------
When all tests succeed and the docs are fine, start a release branch.
**Do not touch any actual code on this branch other than indicated below!**

- Create a branch off of current ``master`` with the name ``release-X.Y.Z``, inserting the correct version number, of course.

  .. code-block:: bash

    git fetch -p origin && git checkout origin/master
    git checkout -b release-X.Y.Z
    git push -u my_fork release-X.Y.Z

- **Important:** This branch will *not* be merged into ``master`` later, thus it does not make sense to create a PR from it.


.. _dev_rel_bump_master:

4. Bump the ``master`` branch to the next development version
-------------------------------------------------------------
To ensure a higher version number for installations from the git master branch, the version number must be increased to a higher value than the upcoming release.

- On the ``master`` branch, change the version string in ``odl/__init__.py`` to the next revision larger than the upcoming release version (or whatever version you know will come next), plus ``'dev0'``.
  For example, if the release version string is ``'0.5.3'``, use ``'0.5.4.dev0'``.

  To make sure you don't miss any other location (or the information here is outdated), perform a search:

  .. code-block:: bash

    cd doc && make clean && cd ..  # remove the local HTML doc first
    grep -Ir "0\.5\.4" . | grep -E -v "\.git|release_notes\.rst|odl\.egg-info"

- In the file ``conda/meta.yaml``, change the version string after ``version:`` to the same as above, but without the ``0`` at the end.
  In the example above, this would mean to change it from ``"0.5.3"`` to ``"0.5.4.dev"``.
  We omit the number since ``conda`` has its own system to enumerate build numbers.

  If necessary, change ``git_rev`` value to ``master``, although that should already be the case.

- Make sure that building packages with ``conda`` still works (see :ref:`Section 6<dev_rel_create_pkgs>` for details).
  If changes to the build system are necessary, test and deploy them in this phase so that building packages on the release branch goes smoothly later on.
- Commit the changes, using a message like ``REL: bump version to X.Y.Z.dev0``.
- Make a PR and merge it after review.


.. _dev_rel_publish:

5. Compile and publish the release
----------------------------------
It is now time to prepare the release documents, increment the version number and make a release on GitHub.
The most important points to keep in mind here are:

Do **not** merge the release branch!

The *only* changes on the release branch should be the version number changes detailed below, nothing else!

Be *very* paranoid and double-check that the version tag under ``git_rev`` in the ``meta.yaml`` file matches **exactly** the tag used on the GitHub release page.
If there is a mismatch, ``conda`` packages won't build, and fixing the situation will be tedious.

.. note::
    The release notes should actually be a running document where everybody who files a PR also makes an entry into the release notes file.
    If not, tough on you -- it is your duty now to make up for all that missed work.
    Maybe you'll remind your co-workers to do this in their next PR.

- Compile the release notes.
  They should contain all *user-visible* changes, including performance improvements and other niceties -- internal stuff like test modifications don't belong here.
  The changes should be summarized in one or two sentences on top, perhaps mentioning the most notable ones in a separate *Highlights* section.
  Check the `Release Notes <https://github.com/odlgroup/odl/blob/master/doc/source/release_notes.rst>`_ file for details on sections, formatting etc.
- Increment the version number in ``odl/__init__.py`` and ``conda/meta.yaml``.
  As in :ref:`Section 4<dev_rel_bump_master>`, perform a search to make sure you didn't miss a version info location.
- Change the ``git_rev`` field in ``conda/meta.yaml`` to ``'vX.Y.Z'``, using the upcoming version number.
  This is the git tag you will create when making the release on GitHub.
- Commit the changes, using a message like ``REL: bump version to X.Y.Z``.
- These changes should *absolutely* be the only ones on the release branch.
- Push the release branch to the main repository so that it is possible to make a `GitHub release <https://github.com/odlgroup/odl/releases>`_ from it:

  .. code-block:: bash

    git push origin release-X.Y.Z

- Go to the `Releases <https://github.com/odlgroup/odl/releases>`_ page on GitHub.
  Click on *Draft a new release* and **select the** ``release-X.Y.Z`` **branch from the dropdown menu, not master**.
  Use ``vX.Y.Z`` as release tag (numbers inserted, of course).
- Paste the short summary (and highlights if written down) from the release notes file (converting from RST to Markdown) but don't insert the details.
- Add a link to the `release notes documentation page <https://odlgroup.github.io/odl/release_notes.html>`_, as in earlier releases.
  Later on, when the documentation with the new release notes is online, you can edit this link to point to the exact section.

.. note::

    If you encounter an issue (like a failing test) that needs immediate fix, stop at that point, fix the issue on a branch *off of* ``master``, make a PR and merge it into ``master`` after review.
    After that, rebase the release branch(es) on the new master and continue.

.. _dev_rel_create_pkgs:

6. Create packages for PyPI and Conda
-------------------------------------
The packages should be built on the release branch to make sure that the version information is correct.

- Making the packages for PyPI is straightforward.
  However, **make sure you delete old** ``build`` **directories** since they can pollute new builds:

  .. code-block:: bash

    rm build/ -rf
    python setup.py sdist
    python setup.py bdist_wheel

  The packages are by default stored in a ``dist`` folder.

- To build the conda packages, you should *not* work in a specific environment but rather exit to the root environment.
  There, install the ``conda-build`` tool for building packages:

  .. code-block:: bash

    conda deactivate
    conda install conda-build

- Invoke the following command to build a package for your platform and all supported Python versions:

  .. code-block:: bash

    conda build conda/ --python 2.7
    conda build conda/ --python 3.5
    conda build conda/ --python 3.6
    conda build conda/ --python 3.7
    ...

- Assuming this succeeds, enter the directory one above where the conda package was stored (as printed in the output).
  For example, if the package was stored as ``$HOME/miniconda3/conda-bld/linux-64/odl-X.Y.Z-py36_0.bz2``, issue the command

  .. code-block:: bash

    cd $HOME/miniconda3/conda-bld/

  In this directory, for each Python version "translate" the package to all platforms since ODL is actually platform-independent:

  .. code-block:: bash

    conda convert --platform all <package>

  Replace ``<package>`` by the package file as built by the previous ``conda build`` command.


.. _dev_rel_test_pkgs:

7. Test installing the PyPI packages and check them
---------------------------------------------------
Before actually uploading packages to "official" servers, first install the local packages and run the unit tests.
Since ``conda-build`` already does this while creating the packages, we can focus on the PyPI packages here.

- Install directly from the source package (``*.tar.gz``) or the wheel (``*.whl``) into a new conda environment:

  .. code-block:: bash

    conda deactivate
    conda create -n pypi_install pytest python=X.Y  # choose Python version
    conda activate pypi_install
    cd /path/to/odl_repo
    cd dist
    pip install <pkg_filename>
    python -c "import odl; odl.test()"

  .. warning::

    Make sure that you're not in the repository root directory while testing, since this can confuse the ``import odl`` command.
    The installed package should be tested, not the code repository.


.. _dev_rel_upload_pkgs:

8. Upload the packages to the official locations
------------------------------------------------
Installing the packages works, now it's time to put them out into the wild.

- Install the ``twine`` package for uploading packages to PyPI in your working environment:

  .. code-block:: bash

    conda deactivate
    conda activate release36
    conda install twine

- Upload the source package and the wheel to the PyPI server using ``twine``:

  .. code-block:: bash

    cd /path/to/odl_repo
    twine upload -u odlgroup dist/<pkg_filename>

  This requires the access credentials for the ``odlgroup`` user on PyPI -- the maintainers have them.

- Upload the conda packages to the ``odlgroup`` channel in the Anaconda cloud.
  The upload requires the ``anaconda-client`` package:

  .. code-block:: bash

    conda install anaconda-client
    cd $HOME/miniconda3/conda-bld
    anaconda upload -u odlgroup `find . -name "odl-X.Y.Z*"`

  For this step, you need the access credentials for the ``odlgroup`` user on the Anaconda server.
  Talk to the maintainers to get them.

.. _dev_rel_merge_release_pr:


Done!
-----
Time to clean up, i.e., remove temporary conda environments, run ``conda build purge``, remove files in ``dist`` and ``build`` generated for the PyPI packages, etc.
