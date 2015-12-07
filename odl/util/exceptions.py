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

"""ODL specific exceptions."""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super


__all__ = ('OpTypeError', 'OpDomainError', 'OpRangeError',
           'OperatorNotImplementedError', 'LinearSpaceTypeError',
           'LinearSpaceNotImplementedError')


class OpTypeError(TypeError):
    """Exception for operator type errors.

    Domain errors are raised by `Operator` subclasses when trying to call
    them with input not in the domain (`Operator.domain`) or with the wrong
    range (`Operator.range`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OpDomainError(OpTypeError):
    """Exception for domain errors.

    Domain errors are raised by `Operator` subclasses when trying to call
    them with input not in the domain (`Operator.domain`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OpRangeError(OpTypeError):
    """Exception for domain errors.

    Domain errors are raised by `Operator` subclasses when the returned
    value does not lie in the range (`Operator.range`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class OperatorNotImplementedError(NotImplementedError):
    """Exception for not implemented errors in `LinearSpace`'s.

    These are raised when a method in `LinearSpace` that has not been
    defined in a specific space is called.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LinearSpaceTypeError(TypeError):
    """Exception for type errors in `LinearSpace`'s.

    These are raised when the wrong type of element is fed to
    `LinearSpace.lincomb` and related functions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LinearSpaceNotImplementedError(NotImplementedError):
    """Exception for not implemented errors in `LinearSpace`'s.

    These are raised when a method in `LinearSpace` that has not been
    defined in a specific space is called.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
