How to contribute
=================

You're reading this, so you actually so you actually want to contribute. Good!

This is some quick information to get you started. At the bottom you find a list of useful links for further reference.

As a user
---------
If you are using the library and find a bug in the code, an error in the [documentation](https://odl.readthedocs.org/index.html), or simply have a question:

Open the [issue tracker](https://github.com/odlgroup/odl/issues), search and check if there is already an issue on your topic and, if not, write a new one! If you do so, please

- be specific,
- be concise,
- provide a [Minimal Working (or rather failing) Example](https://stackoverflow.com/help/mcve),
- point to a code location if possible,
- [send a patch](https://odl.readthedocs.org/dev/gitwash/patching.html) if you have a fix.

As a developer
--------------
To just get acquainted to the code, [clone the repository](https://odl.readthedocs.org/dev/gitwash/following_latest.html) -- you don't need a GitHub account for this step. A good starting point are the [examples](https://github.com/odlgroup/odl/tree/master/examples).

If you want to implement a new great feature, create [your own fork](https://odl.readthedocs.org/dev/gitwash/forking_hell.html#forking) of the repository (requires a GitHub account). You may want to read all of the [development section](https://odl.readthedocs.org/dev/dev.html) in the documentation.

Once you are done with your feature,

- write **tests** for it (untested code is buggy code),
- **document** your new shiny code -- we follow the [NumPy/Scipy style](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt),
- check the [coding style](http://www.python.org/dev/peps/pep-0008/) -- the great tool [autopep8](https://pypi.python.org/pypi/autopep8/) will help you save time,
- when you're ready, [send a pull request](https://odl.readthedocs.org/dev/gitwash/development_workflow.html#ask-for-your-changes-to-be-reviewed-or-merged) -- we'll check your code and get back to you.

Ownership of Code
-----------------
The ownership of your code remains with **you**, as documented by the commits in the Git history. You don't need to sign any [CLA](https://en.wikipedia.org/wiki/Contributor_License_Agreement) whatsoever, and there's no assignment of copyright.

Great tools
-----------
Here's a short list of tools which make life as a developer much easier:

- [Spyder](https://github.com/spyder-ide/spyder) -- a great development environment
- [Pylint](http://www.pylint.org/) -- angry code checker (integrated in Spyder)
- [autopep8](https://pypi.python.org/pypi/autopep8) -- automatically reformat your source according to [PEP8](https://www.python.org/dev/peps/pep-0008/)
- [pytest](https://pytest.org/) -- run unit tests, including [PEP8](https://www.python.org/dev/peps/pep-0008/) style checking using [`pytest --pep8`](https://pypi.python.org/pypi/pytest-pep8)
- [pep257](https://pypi.python.org/pypi/pep257) -- check if your docstrings are in line with [PEP257](https://www.python.org/dev/peps/pep-0257/)
- [GitHub markdown](https://guides.github.com/features/mastering-markdown/) -- helps you write nice and well-structured texts on the issue tracker
