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

"""Run all doctests."""

import nose
import sys

arg = sys.argv[:1]
arg.append('--verbosity=2')
arg.append('--with-coverage')
arg.append('--with-doctest')
arg.append('--doctest-options=+NORMALIZE_WHITESPACE')
try:
    import RL.space.cuda
except ImportError:
    arg.append('--ignore-files=cuda.py')
out = nose.run(defaultTest='./RL/.', argv=arg)
