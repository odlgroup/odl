# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Setup script for ODL.

Installation command::

    pip install [--user] [-e] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import os
import sys


if os.environ.get('READTHEDOCS', None) == 'True':
    # Mock requires in conf.py
    requires = ''
    test_requires = []
else:
    requires = open(
        os.path.join(os.path.dirname(__file__),
                     'requirements.txt')).readlines()
    test_requires = open(
        os.path.join(os.path.dirname(__file__),
                     'test_requirements.txt')).readlines()


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

long_description = """
Operator Discretization Library (ODL) is a Python library for fast prototyping focusing on (but not restricted to) inverse problems. ODL is being developed at `KTH Royal Institute of Technology <https://www.kth.se/en/sci/institutioner/math>`_.

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
- Support for tomographic imaging with a unified geometry representation and bindings to external libraries for efficient computation of projections and back-projections.
- Standardized tests to validate implementations against expected behavior of the corresponding mathematical object, e.g. if a user-defined norm satisfies `norm(x + y) <= norm(x) + norm(y)` for a number of input vectors `x` and `y`.
"""

setup(
    name='odl',

    version='0.5.0',

    description='Operator Discretization Library',
    long_description=long_description,

    url='https://github.com/odlgroup/odl',

    author='ODL development group',
    author_email='odl@math.kth.se',

    license='GPLv3+',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',

        'Operating System :: OS Independent'
    ],

    keywords='research development mathematics prototyping imaging tomography',

    packages=find_packages(exclude=['*test*']),
    package_dir={'odl': 'odl'},

    install_requires=[requires],
    tests_require=['pytest'],
    extras_require={
        'testing': test_requires,
        'show': 'matplotlib',
        'fftw': 'pyfftw',
        'pywavelets': 'Pywavelets>=0.4',
        'scikit': 'scikit-image',
        'proximal': 'proximal',
    },

    cmdclass={'test': PyTest},
)
