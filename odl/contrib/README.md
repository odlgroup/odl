# The `odl.contrib` sub-package

This is a special place for user-contributed modules. Its intention is to provide a "fast lane" for you as a contributor to include code that is

- outside the scope of the main ODL library,
- experimental and untested,
- still subject to major breaking changes or
- "not quite there yet" in terms of documentation, testing or completeness.

The general idea is that you get your code quickly merged without *having to* meet the quality and consistency standards of main ODL that we ensure through extended code reviews. The reasons why we still want your contributed code to be part of the main ODL *repository* are (1) that ODL is intended to follow a ["batteries included"](https://www.python.org/dev/peps/pep-0206/#batteries-included-philosophy) philosophy, and (2) to give you as a contributor maximum visibility.

**Important:** Responsibility for code that you contribute will remain with you. This means in particular that issues on the tracker about your module will be assigned to you. Of course, others will try to help if necessary.

**Note:** If you consider your code to be useful for the main library, please consider filing a pull request anyway. We try to be as helpful as possible in our reviews and make suggestions, so that getting the code ready for merging doesn't take too long.
Note also that contributed modules that attract significant attention and user adoption will be considered for core ODL.

Here are some guidelines for contributors regarding the scope and the rules of this platform.


## Scope

To determine for yourself if your code fits into `odl.contrib`, consult the following checklist:

**Yes:**

- Generic code that would fit into main ODL but is still experimental,
- Special-purpose code that is not generic enough to be a fit for main ODL (e.g. [figures of merit](https://github.com/odlgroup/odl/pull/1018) or application-specific code),
- Extended, more complex examples than [the ones in main ODL](https://github.com/odlgroup/odl/tree/master/examples).

**No:**

- Code for specialized algorithms written specifically for your latest paper,
- Large files like images, datasets etc., or huge amounts of code with limited usefulness,
- Code under licenses other than MPL-2.0.

For code of the type "your latest paper", we recommend using a separate Github repository like [this one](https://github.com/kohr-h/variable_lp_paper) and making a pull request to add your project to the [Applications](https://github.com/odlgroup/odl#applications) list in the main README.


## Procedure

Since you do not have write access to the target location in the ODL repository, do the initial work on the code on [your personal fork](https://odl.readthedocs.io/dev/gitwash/forking_hell.html#forking). Within this fork, add your contribution to the `odl/contrib` directory. When the code is ready, the steps are as follows:

- Submit a pull request. If you would like an in-depth review, please indicate so in writing or by choosing a reviewer on the Github pull request page.
- We will check the pull request as a whole and (optionally) the code. Note that our Travis CI will not check anything in `odl.contrib`.
- To keep the history clean and to avoid accidental bloating of the repository, we will squash your pull request into a single commit that summarizes the whole addition. You can choose to do that yourself and write [a good commit message](https://odl.readthedocs.io/dev/gitwash/development_workflow.html#the-commit-message).
- If no in-depth review was asked for, we will simply merge the PR if everything is okay. Otherwise, merging is delayed until you indicate that you have modified the code according to review.

That's all.


## Package layout

All code goes into a subdirectory, e.g. `mylib`, of the `odl/contrib` directory.
If you have code examples, we strongly suggest that you put them into an `examples` directory in `odl/contrib/mylib/`. The reason for choosing this specific name is that any directory named `examples` is automatically ignored when running `pytest` (see below), which ensures that examples that open new windows, read from disk or take long time to run do not interfere with test runs.
For unit tests, we recommend that you put them into `odl/contrib/mylib/test` or similar.

In summary, the structure after adding your code would look like this:
```
 |
 |---- odl
 |      |
 |      |-- contrib
 |      |       |
 |      ...     |-- __init__.py
 |              |
 |              |-- mylib
 |              ...    |
 |                     |-- __init__.py
 |                     |-- mymodule1.py
 |                     |-- mymodule2.py
 |                     |
 |                     |-- examples
 |                     |      |
 |                     |      |-- example1.py
 |                     |      |-- example2.py
 |                     |      ...
 |                     |
 |                     |--- tests
 |                     ...    |
 |                            |-- test_mymodule1.py
 |                            |-- test_mymodule2.py
 ...                          ...
```
Your package will then be importable as `from odl.contrib import mylib`.

**Important:** The top-level `odl/contrib/__init__.py` should *never* be modified. In particular, it should not import any modules by default. This ensures that code in a submodule does not break or slow down the import of ODL as a whole.


## Tests

We recommend that you add [doctests](https://odlgroup.github.io/odl/dev/testing.html#doctests) to your code from the beginning. They are a natural way of showing to users examples of your code in action, while acting as small unit tests. For complex functionality, you should consider adding [unit tests](https://odlgroup.github.io/odl/dev/testing.html#unit-tests).

**Note:** If you add unit tests, they will by default *not* be run together with the rest of the ODL test suite. This relaxes requirements regarding success and runtime somewhat, but keep in mind that users (and you) will only want to run the tests if they pass and do not take too much time.

To test all of your code with [pytest](https://pytest.org/), run `pytest odl/contrib/mylib/` for the unit tests only and `pytest odl/contrib/mylib --doctest-modules` to also run the doctests.
Also check your code for [PEP8](https://www.python.org/dev/peps/pep-0008/) compliance via `pytest --pep8`. For more information, see the [ODL testing documentation](https://odlgroup.github.io/odl/dev/testing.html).
