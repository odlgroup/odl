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

Install usage:
>>> python setup.py install
"""

from __future__ import print_function, absolute_import

from setuptools import setup, find_packages

__version__ = '0.9b1'

requires = """
future >= 0.14
numpy >= 1.8
scipy >= 0.14
nose >= 1.3
"""

setup(name='odl',
      version=__version__,
      author='ODL development group',
      author_email='kohr@kth.se, jonasadl@kth.se',
      url='https://gits-14.sys.kth.se/LCR/ODL',
      description='Operator Discretization Library',
      license='GPLv3',
      packages=find_packages(exclude=['*test*']),
      install_requires=[requires],
      # packages=['odl', 'odl.discr', 'odl.operator',
      #           'odl.space', 'odl.util'],
      package_dir={'odl': 'odl'})
