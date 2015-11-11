# Copyright 2014, 2015 The ODL development group
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

Installation command:

	pip install [--user] [--editable] .
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import os

if os.environ.get('READTHEDOCS', None) == 'True':
    if sys.version_info < (3, 3):
        from mock import Mock as MagicMock
    else:
        from unittest.mock import MagicMock

    class Mock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
                return Mock()

    MOCK_MODULES = ['scipy', 'numpy', 'odlpp']
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)


__version__ = '0.9b1'


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
        import sys
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

requires = """
future >= 0.14
numpy >= 1.8
scipy >= 0.14
"""

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
          'testing': [
              'pytest >= 2.8.0',
              'coverage >= 4.0'
              ]
      })
