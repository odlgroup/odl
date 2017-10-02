# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Setup script for ODL.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import os
import sys


root_path = os.path.dirname(__file__)


requires = open(os.path.join(root_path, 'requirements.txt')).readlines()
test_requires = open(
    os.path.join(root_path, 'test_requirements.txt')).readlines()


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


test_path = os.path.join(root_path, 'odl', 'test')


def find_tests():
    """Discover the test files for packaging."""
    tests = []
    for path, _, filenames in os.walk(os.path.join(root_path, test_path)):
        for filename in filenames:
            basename, suffix = os.path.splitext(filename)
            if (suffix == '.py' and
                    (basename.startswith('test_') or
                     basename.endswith('_test'))):
                tests.append(os.path.join(path, filename))

    return tests


# Determine version from top-level package __init__.py file
with open(os.path.join(root_path, 'odl', '__init__.py')) as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break


long_description = """
Operator Discretization Library (ODL) is a Python library for fast prototyping focusing on (but not restricted to) inverse problems. ODL is being developed at `KTH Royal Institute of Technology, Stockholm <https://www.kth.se/en/sci/institutioner/math>`_, and `Centrum Wiskunde & Informatica (CWI), Amsterdam <https://www.cwi.nl>`_.

The main intent of ODL is to enable mathematicians and applied scientists to use different numerical methods on real-world problems without having to implement all necessary parts from the bottom up.
This is reached by an `Operator` structure which encapsulates all application-specific parts, and a high-level formulation of solvers which usually expect an operator, data and additional parameters.
The main advantages of this approach is that

1. Different problems can be solved with the same method (e.g. TV regularization) by simply switching operator and data.
2. The same problem can be solved with different methods by simply calling into different solvers.
3. Solvers and application-specific code need to be written only once, in one place, and can be tested individually.
4. Adding new applications or solution methods becomes a much easier task.



Features
========

- Efficient and well-tested data containers based on Numpy (default) or CUDA (optional)
- Objects to represent mathematical notions like vector spaces and operators, including properties as expected from mathematics (inner product, norm, operator composition, ...)
- Convenience functionality for operators like arithmetic, composition, operator matrices etc., which satisfy the known mathematical rules.
- Out-of-the-box support for frequently used operators like scaling, partial derivative, gradient, Fourier transform etc.
- A versatile and pluggable library of optimization routines for smooth and non-smooth problems, such as CGLS, BFGS, Chambolle-Pock and Douglas-Rachford splitting.
- Support for tomographic imaging with a unified geometry representation and bindings to external libraries for efficient computation of projections and back-projections.
- Standardized tests to validate implementations against expected behavior of the corresponding mathematical object, e.g. if a user-defined norm satisfies `norm(x + y) <= norm(x) + norm(y)` for a number of input vectors `x` and `y`.
"""

setup(
    name='odl',

    version=version,

    description='Operator Discretization Library',
    long_description=long_description,

    url='https://github.com/odlgroup/odl',

    author='ODL development group',
    author_email='odl@math.kth.se',

    license='MPL-2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',

        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',

        'Operating System :: OS Independent'
    ],

    keywords='research development mathematics prototyping imaging tomography',

    packages=find_packages(),
    package_dir={'odl': 'odl'},
    package_data={'odl': find_tests() + ['odl/pytest.ini']},
    include_package_data=True,

    install_requires=[requires],
    tests_require=['pytest'],
    extras_require={
        'testing': test_requires,
        'show': 'matplotlib',
        'fftw': 'pyfftw',
        'pywavelets': 'Pywavelets>=0.4',
        'skimage': 'scikit-image',
        'proximal': 'proximal',
    },

    cmdclass={'test': PyTest},
)
