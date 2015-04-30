# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


# from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()


from distutils.core import setup

# from RL import __version__

setup(name='RL',
      # version=__version__,
      version='0.01a',
      author='Holger Kohr, Jonas Adler',
      author_email='kohr@kth.se, jonasadl@kth.se',
      url='https://gits-14.sys.kth.se/LCR/RL',
      description='Regularization Library',
      license='GPLv3',
      packages=['RL', 'RL.geometry', 'RL.operator', 'RL.space', 'RL.utility'],
      package_dir={'RL': 'RL'})
