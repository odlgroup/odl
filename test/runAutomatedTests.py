# -*- coding: utf-8 -*-
"""
simple_test_astra.py -- a simple test script

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest

#Runs all automated tests

from operator_test import *
from linear_operator_test import *
from productpace_test import *
from rn_test import *
from functionSpaces_test import *
from defaultSolvers_test import *

try: #Only run these tests if RLCpp is available
    from cudarn_test import *
    from difference_test_cuda import *
except ImportError:
    print("Could not run cuda tests, lacking RLCpp")

if __name__ == '__main__':
    unittest.main(exit=False)
