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

    pip install [--user] [--E] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import os
import sys

__version__ = '0.9b1'

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

setup(name='odl',
      version=__version__,
      author='ODL development group',
      author_email='kohr@kth.se, jonasadl@kth.se',
      url='https://github.com/odlgroup/odl',
      description='Operator Discretization Library',
      license='GPLv3',
      packages=find_packages(exclude=['*test*']),
      install_requires=[requires],
      package_dir={'odl': 'odl'},
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      extras_require={
          'testing': test_requires,
          'wavelets': 'Pywavelets',
          'fft': 'pyfftw',
          'show': 'matplotlib',
          'all': ['Pywavelets', 'pyfftw', 'matplotlib']
      })
