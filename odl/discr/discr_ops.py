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

"""Operators defined on `DiscreteLp`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.discr import DiscreteLp, uniform_discr
from odl.operator import Operator
from odl.util.normalize import normalized_scalar_param_list, safe_int_conv
from odl.util.numerics import resize_array, _SUPPORTED_PAD_MODES


__all__ = ('Resampling', 'ResizingOperator')


class Resampling(Operator):

    """An operator that resamples on a different grid in the same set.

    The operator uses the underlying `DiscretizedSet.sampling` and
    `DiscretizedSet.interpolation` operators to achieve this.

    The spaces need to have the same `DiscretizedSet.uspace` in order
    for this to work. The data space implementations may be different,
    although performance may suffer drastically due to translation
    steps.
    """

    def __init__(self, domain, range):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscretizedSet`
            Set of elements that are to be resampled.
        range : `DiscretizedSet`
            Set in which the resampled elements lie.

        Examples
        --------
        Create two spaces with different number of points and a resampling
        operator.

        >>> import odl
        >>> coarse_discr = odl.uniform_discr(0, 1, 3)
        >>> fine_discr = odl.uniform_discr(0, 1, 6)
        >>> resampling = odl.Resampling(coarse_discr, fine_discr)
        >>> resampling.domain
        uniform_discr(0.0, 1.0, 3)
        >>> resampling.range
        uniform_discr(0.0, 1.0, 6)
        """
        if domain.uspace != range.uspace:
            raise ValueError('`domain.uspace` ({}) does not match '
                             '`range.uspace` ({})'
                             ''.format(domain.uspace, range.uspace))

        super().__init__(domain=domain, range=range, linear=True)

    def _call(self, x, out=None):
        """Apply resampling operator.

        The vector ``x`` is resampled using the sampling and interpolation
        operators of the underlying spaces.

        Examples
        --------
        Create two spaces with different number of points and apply the
        corresponding resampling operator to an element:

        >>> import odl
        >>> coarse_discr = odl.uniform_discr(0, 1, 3)
        >>> fine_discr = odl.uniform_discr(0, 1, 6)
        >>> resampling = odl.Resampling(coarse_discr, fine_discr)
        >>> print(resampling([0, 1, 0]))
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

        The result depends on the interpolation chosen for the underlying
        spaces:

        >>> Z = odl.uniform_discr(0, 1, 3, interp='linear')
        >>> linear_resampling = Resampling(Z, Y)
        >>> print(linear_resampling([0, 1, 0]))
        [0.0, 0.25, 0.75, 0.75, 0.25, 0.0]
        """
        if out is None:
            return x.interpolation
        else:
            out.sampling(x.interpolation)

    @property
    def inverse(self):
        """An (approximate) inverse of this resampling operator.

        The returned operator is resampling defined in the opposite
        direction.

        See Also
        --------
        adjoint : resampling is unitary, so the adjoint is the inverse.
        """
        return Resampling(self.range, self.domain)

    @property
    def adjoint(self):
        """Return an (approximate) adjoint.

        The result is only exact if the interpolation and sampling
        operators of the underlying spaces match exactly.

        Returns
        -------
        adjoint : Resampling
            Resampling operator defined in the opposite direction.

        Examples
        --------
        Create resampling operator and inverse:

        >>> import odl
        >>> coarse_discr = odl.uniform_discr(0, 1, 3)
        >>> fine_discr = odl.uniform_discr(0, 1, 6)
        >>> resampling = odl.Resampling(coarse_discr, fine_discr)
        >>> resampling_inv = resampling.inverse

        The inverse is proper left inverse if the resampling goes from a
        coarser to a finer sampling:

        >>> x = [0.0, 1.0, 0.0]
        >>> print(resampling_inv(resampling(x)))
        [0.0, 1.0, 0.0]

        But can fail in the other direction

        >>> y = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        >>> print(resampling(resampling_inv(y)))
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        """
        return self.inverse


class ResizingOperator(Operator):

    """Operator mapping between discretized spaces of different shapes.

    This operator is intended as a mapping between uniformly
    discretized spaces with the same `DiscreteLp.cell_sides` but
    different `DiscreteLp.shape`. The underlying operation is array
    resizing, i.e. no resampling is performed.

    By default, the `Operator.range` is a uniformly discretized space
    with  the same properties as `Operator.domain`, except for changed
    shape.

    All resizing operator variants are linear, except constant padding
    with constant != 0.
    """

    def __init__(self, domain, range=None, ran_shp=None, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        domain : uniform `DiscreteLp`
            Uniformly discretized space, the operator can be applied
            to its elements.
        range : uniform `DiscreteLp`, optional
            Uniformly discretized space in which the result of the
            application of this operator lies.
            For the default ``None``, a space with the same attributes
            as ``domain`` is used, except for its shape, which is set
            to ``ran_shp``.
        ran_shp : sequence of int
            Shape of the range of this operator. This can be provided
            instead of ``range`` and is mandatory if ``range`` is
            ``None``.
        num_left : int or sequence of int, optional
            Number of cells to add to/remove from the left of
            ``domain.partition``. By default, the difference is
            distributed evenly, with preference for left in case of
            ambiguity.
            This option is can only be used together with ``ran_shp``.
        pad_mode : str, optional
            Method to be used to fill in missing values in an enlarged array.

            ``'constant'``: Fill with ``pad_const``.

            ``'symmetric'``: Reflect at the boundaries, not doubling the
            outmost values. This requires left and right padding sizes
            to be strictly smaller than the original array shape.

            ``'periodic'``: Fill in values from the other side, keeping
            the order. This requires left and right padding sizes to be
            at most as large as the original array shape.

            ``'order0'``: Extend constantly with the outmost values
            (ensures continuity).

            ``'order1'``: Extend with constant slope (ensures continuity of
            the first derivative). This requires at least 2 values along
            each axis where padding is applied.

            Note that for ``'symmetric'`` and ``'periodic'`` padding, the
            number of added values on each side of the array cannot exceed
            the original size.

        pad_const : scalar, optional
            Value to be used in the ``'constant'`` padding mode.

        discr_kwargs: dict, optional
            Keyword arguments passed to the `uniform_discr` constructor.
        """
        if not isinstance(domain, DiscreteLp):
            raise TypeError('`domain` must be a `DiscreteLp` instance, '
                            'got {!r}'.format(domain))

        if not domain.is_uniform:
            raise ValueError('`domain` is not uniformly discretized')

        num_left = kwargs.pop('num_left', None)
        discr_kwargs = kwargs.pop('discr_kwargs', {})

        if range is None:
            if ran_shp is None:
                raise ValueError('either `range` or `ran_shp` must be '
                                 'given')

            num_left = normalized_scalar_param_list(
                num_left, domain.ndim, param_conv=safe_int_conv)
            self.__num_left = tuple(num_left)

            range = _resize_discr(domain, ran_shp, num_left, discr_kwargs)

        elif ran_shp is None:
            if num_left is not None:
                raise ValueError('`num_left` can only be combined with '
                                 '`ran_shp`')

            if not np.allclose(range.cell_sides, domain.cell_sides):
                raise ValueError(
                    'cell sides of domain and range differ significantly '
                    '(difference {})'
                    ''.format(range.cell_sides - domain.cell_sides))

            self.__num_left = self._num_left_from_spaces(domain, range)

        else:
            raise ValueError('cannot combine `range` with `ran_shape`')

        pad_mode = kwargs.pop('pad_mode', 'constant')
        self.__pad_mode = str(pad_mode).lower()
        if self.pad_mode not in _SUPPORTED_PAD_MODES:
            raise ValueError("`pad_mode` '{}' not understood".format(pad_mode))

        self.__pad_const = float(kwargs.pop('pad_const', 0.0))

        # padding mode 'constant' with `pad_const != 0` is not linear
        linear = (self.pad_mode != 'constant' or self.pad_const == 0.0)

        super().__init__(domain, range, linear=linear)

    @property
    def num_left(self):
        """Number of cells added to/removed from the left."""
        return self.__num_left

    @property
    def pad_mode(self):
        """Padding mode used by this operator."""
        return self.__pad_mode

    @property
    def pad_const(self):
        """Constant used by this operator in case of constant padding."""
        return self.__pad_const

    def _call(self, x, out):
        """Implement ``self(x, out)``."""
        # TODO: simplify once context manager is available
        out[:] = resize_array(x.asarray(), self.range.shape,
                              num_left=self.num_left, pad_mode=self.pad_mode,
                              pad_const=self.pad_const, out=out.asarray())

    def derivative(self, point):
        """Derivative of this operator.

        For the particular case of constant padding with non-zero
        constant, the derivative is the corresponding zero-padding
        variant. In all other cases, this operator is linear, i.e.
        the derivative is equal to ``self``.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return ResizingOperator(
                domain=self.domain, range=self.range, pad_mode='constant',
                pad_const=0.0)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        In axes where ``self`` extends, the adjoint is given by the
        corresponding restriction. In restriction axes, the adjoint
        performs zero-padding.
        """
        if not self.is_linear:
            raise NotImplementedError('this operator is not linear and '
                                      'thus has no adjoint')
        return ResizingOperator(domain=self.range, range=self.domain,
                                pad_mode='constant', pad_const=0.0)

    # TODO: inverse

    @staticmethod
    def _num_left_from_spaces(dom, ran):
        """Return num_left corresponding to given spaces."""
        diff_l = np.abs(ran.grid.min() - dom.grid.min())
        num_left_float = diff_l / dom.cell_sides
        num_left = np.around(num_left_float).astype(int)
        if not np.allclose(num_left, num_left_float):
            raise ValueError('range is shifted relative to domain by a '
                             'non-multiple of cell_sides')
        return tuple(num_left)


def _resize_discr(discr, newshp, num_left, discr_kwargs):
    """Return ``discr`` resized to ``newshp``.

    Resize to ``newshp``, using ``num_left`` added/removed points to
    the left (per axis). In axes where ``num_left`` is ``None``, the
    points are distributed evenly.
    """
    new_begin, new_end = [], []
    for b_orig, e_orig, n_orig, cs, n_new, num_l in zip(
            discr.min_corner, discr.max_corner, discr.shape, discr.cell_sides,
            newshp, num_left):

        n_diff = n_new - n_orig
#        print(n_diff)
        if num_l is None:
            num_r = n_diff // 2
            num_l = n_diff - num_r
        else:
            num_r = n_diff - num_l

#        print(num_l, num_r)

        new_begin.append(b_orig - num_l * cs)
        new_end.append(e_orig + num_r * cs)

#    print(new_begin, new_end)

    return uniform_discr(new_begin, new_end, newshp, **discr_kwargs)

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
