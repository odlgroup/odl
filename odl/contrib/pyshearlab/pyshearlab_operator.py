# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""ODL integration with pyshearlab."""


import odl
import numpy as np
import pyshearlab
from threading import Lock


__all__ = ('PyShearlabOperator',)


class PyShearlabOperator(odl.Operator):

    """Shearlet transform using PyShearlab.

    This is the non-compact shearlet transform implemented using the fourier
    transform.
    """

    def __init__(self, space, num_scales):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp`
            The space on which the shearlet transform should act. Must be
            two-dimensional.
        num_scales : nonnegative `int`
            The number of scales for the shearlet transform, higher numbers
            mean better edge resolution but more computational burden.

        Examples
        --------
        Create a 2d-shearlet transform:

        >>> space = odl.uniform_discr([-1, -1], [1, 1], [128, 128])
        >>> shearlet_transform = PyShearlabOperator(space, num_scales=2)
        """
        self.shearlet_system = pyshearlab.SLgetShearletSystem2D(
            0, space.shape[0], space.shape[1], num_scales)
        range = space ** self.shearlet_system['nShearlets']
        self.mutex = Lock()
        super(PyShearlabOperator, self).__init__(space, range, True)

    def _call(self, x):
        """Return ``self(x)``."""
        with self.mutex:
            result = pyshearlab.SLsheardec2D(x, self.shearlet_system)
            return np.moveaxis(result, -1, 0)

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return PyShearlabOperatorAdjoint(self)

    @property
    def inverse(self):
        """Return the inverse operator."""
        return PyShearlabOperatorInverse(self)


class PyShearlabOperatorAdjoint(odl.Operator):

    """Adjoint of the shearlet transform.

    Should not be used independently.

    See Also
    --------
    odl.contrib.pyshearlab.PyShearlabOperator
    """

    def __init__(self, op):
        """Initialize a new instance.

        Parameters
        ----------
        op : `PyShearlabOperator`
            The operator which this should be the adjoint of.
        """
        self.op = op
        super(PyShearlabOperatorAdjoint, self).__init__(
            op.range, op.domain, True)

    def _call(self, x):
        """Return ``self(x)``."""
        with self.op.mutex:
            x = np.moveaxis(x, 0, -1)
            return pyshearlab.SLshearadjoint2D(x, self.op.shearlet_system)

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return self.op

    @property
    def inverse(self):
        """Return the inverse operator."""
        return PyShearlabOperatorAdjointInverse(self.op)


class PyShearlabOperatorInverse(odl.Operator):

    """Inverse of the shearlet transform.

    Should not be used independently.

    See Also
    --------
    odl.contrib.pyshearlab.PyShearlabOperator
    """

    def __init__(self, op):
        """Initialize a new instance.

        Parameters
        ----------
        op : `PyShearlabOperator`
            The operator which this should be the inverse of.
        """
        self.op = op
        super(PyShearlabOperatorInverse, self).__init__(
            op.range, op.domain, True)

    def _call(self, x):
        """Return ``self(x)``."""
        with self.op.mutex:
            x = np.moveaxis(x, 0, -1)
            return pyshearlab.SLshearrec2D(x, self.op.shearlet_system)

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return PyShearlabOperatorAdjointInverse(self.op)

    @property
    def inverse(self):
        """Return the inverse operator."""
        return self.op


class PyShearlabOperatorAdjointInverse(odl.Operator):

    """Adjoint of the inverse/Inverse of the adjoint of shearlet transform.

    Should not be used independently.

    See Also
    --------
    odl.contrib.pyshearlab.PyShearlabOperator
    """

    def __init__(self, op):
        """Initialize a new instance.

        Parameters
        ----------
        op : `PyShearlabOperator`
            The operator which this should be the inverse of the adjoint of.
        """
        self.op = op
        super(PyShearlabOperatorAdjointInverse, self).__init__(
            op.domain, op.range, True)

    def _call(self, x):
        """Return ``self(x)``."""
        with self.op.mutex:
            result = pyshearlab.SLshearrecadjoint2D(x, self.op.shearlet_system)
            return np.moveaxis(result, -1, 0)

    @property
    def adjoint(self):
        """Return the adjoint operator."""
        return self.op.inverse

    @property
    def inverse(self):
        """Return the inverse operator."""
        return self.op.adjoint


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
