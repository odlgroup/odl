# Release Howto

This document is intended to give precise instructions on the process of making a release. Its purpose is to avoid broken packages, broken documentation and many other things that can go wrong in the release process. Since this is not everyday work and may be done under the stress of a (self-imposed) deadline, it is clearly beneficial to have a checklist to hold on to.

**Note:** The instructions in this document are tentative until tested in practice. They are also written from the perspective of Linux and may need adaption for other platforms.

## 1. Agree on a release schedule

This involves the "what" and "when" of the release process and fixes a feature set that is supposed to be included in the new version. The steps are:

- Open an issue on the issue tracker using the title **Release x.y.z**.
- Discuss and agree on a set of open PRs that should be merged before making a release.

## 2. Make sure tests succeed and docs can be built

When all required PRs are merged, ensure that the latest `master` branch is sane. Travis CI checks every PR, but certain things like CUDA cannot be tested there and must therefore undergo tests on a local machine, for at least Python 2.7 and one version of Python 3.

- Make a new test conda environment and install all dependencies:
  ```
  $ conda create -n release35 python=3.5 nomkl numpy scipy future matplotlib pytest
  $ source activate release35
  $ pip install pyfftw pywavelets
  $ cd /path/to/odl_repo
  $ git checkout master
  $ git pull
  $ pip install -e .
  ```
- Run the tests with `pytest`, including doctests, examples documentation and large-scale tests:
  ```
  $ pytest --doctest-modules --examples --doctest-doc --largescale
  ```
- Do the same for Python 2.7.
- Make sure the tests also run on the platforms you're currently *not* testing on. Ask a buddy maintainer if necessary.
- Build the documentation. This requires `sphinx` and the `sphinxext` submodule:
  ```
  $ conda install sphinx sphinx-rtd-theme
  $ git submodule update --init --recursive
  $ cd doc && make clean
  $ cd source && python generate_doc.py
  $ cd ..
  $ make html 2>&1 |\
    grep -E "SEVERE|ERROR|WARNING" |\
    grep -E -v "more than one target found for|__eq__|document isn't included in any toctree"
  ```
  The last command builds the documentation and filters from the output all irrelevant warnings, letting through only the "proper" warnings and errors. If possible, *fix these remaining issues*.
- Glance the built documentation (usually in `doc/_build`) for obvious errors.

## 3. Make a release branch off `master` and draft the release

When all tests succeed and the docs are fine, start a release branch. All work until the creation of the git release tag is now done on the release branch.
**Do not touch any code on this branch other than indicated below!**

- Create a branch off current `master` with the name `release-X.Y.Z`, inserting the correct version number, of course.
- Compile the release notes. They should contain all *user-visible* changes (internal stuff like modified tests is not required) and should be summarized in one or two sentences on top, perhaps mentioning the most notable changes. Check the [Release Notes](https://github.com/odlgroup/odl/blob/master/doc/source/release_notes.rst) file for details on sections, formatting etc.
- Increment the version number. The current locations of version numbers are the [odl/__init__.py](https://github.com/odlgroup/odl/blob/master/odl/__init__.py) and [conda/meta.yaml](https://github.com/odlgroup/odl/blob/master/conda/meta.yaml). To make sure you don't miss any other location (or the information here is outdated), perform a search:
  ```
  $ cd doc && make clean && cd ..  # remove the local HTML doc first
  $ grep -Ir "0\.5\.4" . | grep -E -v "\.git|release_notes\.rst|odl\.egg-info"
  ```
- Change the `git_rev` field in `conda/meta.yaml` to `'vX.Y.Z'`, using the upcoming version number. This is the git tag you will create when making the release on GitHub.
- Commit the changes, using a message like `REL: bump version to X.Y.Z`.
- Make a PR and merge it after review.

## 4. Make a release on GitHub

Now that the version is incremented,

- make a new [Release](https://github.com/odlgroup/odl/releases) on GitHub.
- Paste the short summary from the release notes file (converting from RST to Markdown) but don't insert the details.
- Add a link to the current section in the release notes file.

## 5. Create packages for PyPI and Conda

- Making the packages for PyPI is straightforward:
  ```
  $ conda install wheel
  $ python setup.py sdist
  $ python setup.py bdist_wheel
  ```
  The packages are by default stored in a `dist` folder.

- To build the conda packages, you should *not* work in a specific environment but rather exit to the root environment. There, install the `conda-build` tool for building packages:
  ```
  $ source deactivate
  $ conda install conda-build
  ```
- Invoke the following command to build a package for your platform and all supported Pyhton versions:
  ```
  $ conda build conda/ --python 2.7
  $ conda build conda/ --python 3.4
  $ conda build conda/ --python 3.5
  ...
  ```
- Assuming this succeeds, enter the directory one above where the conda package was stored (as printed in the output), e.g.,
  ```
  $ cd $HOME/miniconda3/conda-bld/
  ```
  There, for each Python version "translate" the package to all platforms since ODL is actually platform-independent:
  ```
  $ conda convert --platform all <package>
  ```
  Replace `<package>` by the package file as built by the previous `conda build` command.

## 6. Upload the packages to test locations and test installing them

Before actually uploading packages to "official" servers, first use a test location and make sure installation works as expected.

- Install the `twine` package for uploading packages to PyPI in your working environment:
  ```
  $ source activate release35
  $ pip install twine
  ```
- Upload the source package (`*.tar.gz`) and the wheel (`*.whl`) to the test server and try installing it from there into a new conda environment:
  ```
  $ cd /path/to/odl_repo
  $ twine upload -u odlgroup -r pypitest <package>
  $ source deactivate
  $ conda create -n pypi_install python=X.Y  # choose Python version
  $ source activate pypi_install
  $ pip install --index-url https://test.pypi.org/legacy odl
  $ python -c "import odl; odl.test()"
  $ pip uninstall odl
  $ pip install --index-url --no-binary https://test.pypi.org/legacy odl
  $ python -c "import odl; odl.test()"
  ```
  **TODO:** Currently no `odlgroup` user exists on the test server, may need to be created.
- Upload the conda package to *your own* conda channel and test the install procedure in a new conda environment. The upload requires the `anaconda-client` package:
  ```
  $ source deactivate
  $ conda install anaconda-client
  $ anaconda upload -u <your_username> <package1> <package2> <...>
  $ conda create -n conda_install python=X.Y  # choose Python version
  $ source activate conda_install
  $ conda install -c <your_username> odl
  $ python -c "import odl; odl.test()"
  ```

## 7. Upload the packages to the official locations

Installing the packages works, now it's time to put them out into the wild.

- Upload the source package and the wheel to the "real" PyPI server using again `twine`:
  ```
  $ source deactivate && source activate release35
  $ twine upload -u odlgorup -r pypi <package>
  ```
  This requires the access credentials for the `odlgroup` user on PyPI.
- Upload the conda packages to the `odlgroup` channel in the Anaconda cloud:
  ```
  $ cd $HOME/miniconda3/conda-bld
  $ anaconda upload -u odlgroup `find . -name "odl-X.Y.Z*"`
  ```
  For this step, you need the access credentials for the `odlgroup` user on the Anaconda server.

## 8. Bump current `master` to a development version

To ensure a larger version number for installations from the git `master` branch, the version number must be increased immediately.

- Change the version string `'X.Y.Z'` in `odl/__init__.py` to `'X.Y.Z+1.dev0'` (e.g. from `'0.5.3'` to `'0.5.4.dev0'`).
- Change the `git_rev` field in `conda/meta.yaml` to `'master'`.
- Commit the changes, using a message like `REL: bump version to X.Y.Z.dev0`.
- Make a PR with just this change and merge it after review. It should be the first one that goes in after the release.

## Done!

Time to clean up.
