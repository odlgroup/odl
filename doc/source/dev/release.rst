.. _dev_release:

#######################
The ODL release process
#######################

This document is intended to give precise instructions on the process of making a release.
Its purpose is to avoid broken packages, broken documentation and many other things that can go wrong as a result of mistakes during the release process.
Since this is not everyday work and may be done under the stress of a (self-imposed) deadline, it is clearly beneficial to have a checklist to hold on to.

.. note::
    The instructions in this document are tentative until tested in practice.
    They are also written from the perspective of Linux and may need adaption for other platforms.


.. _dev_rel_release_schedule:
1. Agree on a release schedule
------------------------------
This involves the "what" and "when" of the release process and fixes a feature set that is supposed to be included in the new version.
The steps are:

- Open an issue on the issue tracker using the title **Release X.Y.Z** (insert numbers, of course).
- Discuss and agree on a set of open PRs that should be merged before making a release.


.. _dev_rel_master_ok:
2. Make sure tests succeed and docs are built properly
------------------------------------------------------
When all required PRs are merged, ensure that the latest ``master`` branch is sane. Travis CI checks every PR, but certain things like CUDA cannot be tested there and must therefore undergo tests on a local machine, for at least Python 2.7 and one version of Python 3.

- Make a new test conda environment and install all dependencies:

  .. code-block:: bash

    conda create -n release36 python=3.6 nomkl numpy scipy future matplotlib pytest
    source activate release36
    pip install pyfftw pywavelets
    cd /path/to/odl_repo
    git checkout master
    git pull
    pip install -e .

- Run the tests with ``pytest``, including doctests, examples documentation and large-scale tests:

  .. code-block:: bash

    pytest --doctest-modules --examples --doctest-doc --largescale

- Do the same for Python 2.7.
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


.. _dev_rel_release_branch:
3. Make a release branch off ``master``
---------------------------------------
When all tests succeed and the docs are fine, start a release branch.
**Do not touch any actual code on this branch other than indicated below!**

- Create a branch off current ``master`` with the name ``release-X.Y.Z``, inserting the correct version number, of course.

  .. code-block:: bash

    git checkout master
    git pull
    git checkout -b release-X.Y.Z
    git push -u my_fork release-X.Y.Z

  Like any regular branch that should result in a PR, the release branch is pushed to a fork.


.. _dev_rel_bump_master:
4. Bump the ``master`` branch to the next development version
-------------------------------------------------------------
To ensure a larger version number for installations from the git master branch, the version number must be increased before merging the release branch.

- On the ``master`` branch, change the version string in ``odl/__init__.py`` to the next revision larger than the upcoming release version, plus ``'dev0'``.
  For example, if the release version string is ``'0.5.3'``, use ``'0.5.4.dev0'``.

  To make sure you don't miss any other location (or the information here is outdated), perform a search:

  .. code-block:: bash

    cd doc && make clean && cd ..  # remove the local HTML doc first
    grep -Ir "0\.5\.4" . | grep -E -v "\.git|release_notes\.rst|odl\.egg-info"

- In the file ``conda/meta.yaml``, change the version string after ``version: `` to the same as above, but without the ``dev0`` tag.
  In the example above, this would mean to change it from ``"0.5.3"`` to ``"0.5.4"``.

  If necessary, change ``git_rev`` value to ``master``, although that should already be the case.

- Commit the changes, using a message like ``REL: bump version to X.Y.Z.dev0``.
- Make a PR with just this change and merge it after review.
  It must be merged before the release branch.


.. _dev_rel_publish:
5. Compile and publish the release
----------------------------------
Back on the release branch with a ``git checkout release-X.Y.Z``, it is now time to prepare the release documents, increment the version number and make a release on GitHub.

- Compile the release notes.
  They should contain all *user-visible* changes (internal stuff like test modifications is not required) and should be summarized in one or two sentences on top, perhaps mentioning the most notable changes.
  Check the `Release Notes <https://github.com/odlgroup/odl/blob/master/doc/source/release_notes.rst>`_ file for details on sections, formatting etc.
- Increment the version number in ``odl/__init__.py`` and ``conda/meta.yaml``.
  As in :ref:`Section 4<dev_rel_bump_master>`, perform a search to make sure you didn't miss a version info location.
- Change the ``git_rev`` field in ``conda/meta.yaml`` to ``'vX.Y.Z'``, using the upcoming version number.
  This is the git tag you will create when making the release on GitHub.
- Commit the changes, using a message like ``REL: bump version to X.Y.Z``.
- Make a PR and fix review comments.
  When doing so, try to keep the ``REL: bump version to X.Y.Z`` commit last, for example by using ``git commit --amend`` for fixes, or by squashing the commits on the release branch.

  **Don't merge immediately when ready!**

- Make a new `Release <https://github.com/odlgroup/odl/releases>`_ on GitHub **from the release branch, not master**.
- Paste the short summary from the release notes file (converting from RST to Markdown) but don't insert the details.
- Add a link to the current section in the release notes file.


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

    source deactivate
    conda install conda-build

- Invoke the following command to build a package for your platform and all supported Python versions:

  .. code-block:: bash

    conda build conda/ --python 2.7
    conda build conda/ --python 3.4
    conda build conda/ --python 3.5
    conda build conda/ --python 3.6
    ...

- Assuming this succeeds, enter the directory one above where the conda package was stored (as printed in the output), e.g.,

  .. code-block:: bash

    cd $HOME/miniconda3/conda-bld/

  There, for each Python version "translate" the package to all platforms since ODL is actually platform-independent:

  .. code-block:: bash

    conda convert --platform all <package>

  Replace ``<package>`` by the package file as built by the previous ``conda build`` command.


.. _dev_rel_test_pkgs:
7. Test installing the local packages and check them
----------------------------------------------------
Before actually uploading packages to "official" servers, first install the local packages and run the unit tests.

- Install directly from the source package (``*.tar.gz``) or the wheel (``*.whl``) into a new conda environment:

  .. code-block:: bash

    source deactivate
    conda create -n pypi_install python=X.Y  # choose Python version
    source activate pypi_install
    cd /path/to/odl_repo
    pip install dist/<pkg_filename>
    python -c "import odl; odl.test()"

- Install and test the local conda packages in a new conda environment:

  .. code-block:: bash

    source deactivate
    conda create -n conda_install python=X.Y  # choose Python version
    source activate conda_install
    conda install --use-local nomkl odl
    python -c "import odl; odl.test()"


.. _dev_rel_upload_pkgs:
8. Upload the packages to the official locations
------------------------------------------------
Installing the packages works, now it's time to put them out into the wild.

- Install the ``twine`` package for uploading packages to PyPI in your working environment:

  .. code-block:: bash

    source deactivate
    source activate release36
    pip install twine

- Upload the source package and the wheel to the PyPI server using ``twine``:

  .. code-block:: bash

    cd /path/to/odl_repo
    twine upload -u odlgorup dist/<pkg_filename>

  This requires the access credentials for the ``odlgroup`` user on PyPI.
- Upload the conda packages to the ``odlgroup`` channel in the Anaconda cloud.
  The upload requires the ``anaconda-client`` package:

  .. code-block:: bash

    conda install anaconda-client
    cd $HOME/miniconda3/conda-bld
    anaconda upload -u odlgroup `find . -name "odl-X.Y.Z*"`

  For this step, you need the access credentials for the ``odlgroup`` user on the Anaconda server.


.. _dev_rel_merge_release_pr:
9. Merge the release branch
---------------------------
Now the release branch can finally be merged.
The sole purpose of this step is to update the release notes on ``master`` and potentially get the last minute changes.

- The release branch will have conflicts with ``master`` since both have modified the version information.
  Resolve them in favor of the changes made on ``master``.
  In particular, make sure that the changes in :ref:`Section 4<dev_rel_bump_master>` stay intact.
- Merge the PR for the release.

Done!
-----
Time to clean up, i.e., remove temporary conda environments, run ``conda build purge``, remove files in ``dist`` and ``build`` generated for the PyPI packages, etc.
