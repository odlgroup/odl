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


# Imports for common Python 2/3 codebase
from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import
from future import standard_library

# External module imports
import unittest

# Run all automated tests
from operator_test import *
from linear_operator_test import *
from productpace_test import *
from rn_test import *
from functionSpaces_test import *
from defaultSolvers_test import *

try:  # Only run these tests if RLCpp is available
    from cudarn_test import *
    from difference_test_cuda import *
except ImportError:
    print("Could not run cuda tests, lacking RLCpp")

standard_library.install_aliases()


if __name__ == '__main__':
    unittest.main(exit=False)
