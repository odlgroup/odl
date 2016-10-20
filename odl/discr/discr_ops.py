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
from odl.util.numerics import resize_array, _SUPPORTED_RESIZE_PAD_MODES


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

        >>> coarse_discr = odl.uniform_discr(0, 1, 3)
        >>> fine_discr = odl.uniform_discr(0, 1, 6)
        >>> resampling = odl.Resampling(coarse_discr, fine_discr)
        >>> resampling.domain
        uniform_discr(0.0, 1.0, 3)
        >>> resampling.range
        uniform_discr(0.0, 1.0, 6)

        Apply the  corresponding resampling operator to an element:

        >>> print(resampling([0, 1, 0]))
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

        The result depends on the interpolation chosen for the underlying
        spaces:

        >>> coarse_discr = odl.uniform_discr(0, 1, 3, interp='linear')
        >>> linear_resampling = odl.Resampling(coarse_discr, fine_discr)
        >>> print(linear_resampling([0, 1, 0]))
        [0.0, 0.25, 0.75, 0.75, 0.25, 0.0]
        """
        if domain.uspace != range.uspace:
            raise ValueError('`domain.uspace` ({}) does not match '
                             '`range.uspace` ({})'
                             ''.format(domain.uspace, range.uspace))

        super().__init__(domain=domain, range=range, linear=True)

    def _call(self, x, out=None):
        """Apply resampling operator.

        The element ``x`` is resampled using the sampling and interpolation
        operators of the underlying spaces.
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

        >>> coarse_discr = odl.uniform_discr(0, 1, 3)
        >>> fine_discr = odl.uniform_discr(0, 1, 6)
        >>> resampling = odl.Resampling(coarse_discr, fine_discr)
        >>> resampling_inv = resampling.inverse

        The inverse is proper left inverse if the resampling goes from a
        coarser to a finer sampling:

        >>> x = [0.0, 1.0, 0.0]
        >>> print(resampling_inv(resampling(x)))
        [0.0, 1.0, 0.0]

        However, it can fail in the other direction:

        >>> y = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        >>> print(resampling(resampling_inv(y)))
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        """
        return self.inverse


class ResizingOperatorBase(Operator):

    """Base class for `ResizingOperator` and its adjoint.

    This is an abstract class used to share code between the forward and
    adjoint variants of the resizing operator.
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
        ran_shp : sequence of ints, optional
            Shape of the range of this operator. This can be provided
            instead of ``range`` and is mandatory if ``range`` is
            ``None``.
        offset : int or sequence of ints, optional
            Number of cells to add to/remove from the left of
            ``domain.partition``. By default, the difference is
            distributed evenly, with preference for left in case of
            ambiguity.
            This option is can only be used together with ``ran_shp``.
        pad_mode : string, optional
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

        pad_const : scalar, optional
            Value to be used in the ``'constant'`` padding mode.

        discr_kwargs: dict, optional
            Keyword arguments passed to the `uniform_discr` constructor.

        Examples
        --------
        The simplest way of initializing a resizing operator is by
        providing ``ran_shp`` and, optionally, parameters for the padding
        variant to be used. The range is inferred from ``domain`` and
        the supplied parameters. If no ``offset`` is given, the difference
        in size is evenly distributed to both sides:

        >>> space = odl.uniform_discr([0, 0], [1, 1], (2, 4))
        >>> resize_op = odl.ResizingOperator(space, ran_shp=(4, 4))
        >>> resize_op.range
        uniform_discr([-0.5, 0.0], [1.5, 1.0], [4, 4])

        Testing different padding methods in the first axis (zero padding
        is the default):

        >>> x = [[1, 2, 3, 4],
        ...      [5, 6, 7, 8]]
        >>> resize_op = odl.ResizingOperator(space, ran_shp=(4, 4))
        >>> print(resize_op(x))
        [[0.0, 0.0, 0.0, 0.0],
         [1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [0.0, 0.0, 0.0, 0.0]]
        >>>
        >>> resize_op = odl.ResizingOperator(space, ran_shp=(4, 4),
        ...                                  offset=(0, 0),
        ...                                  pad_mode='periodic')
        >>> print(resize_op(x))
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0]]
        >>>
        >>> resize_op = odl.ResizingOperator(space, ran_shp=(4, 4),
        ...                                  offset=(0, 0),
        ...                                  pad_mode='order0')
        >>> print(resize_op(x))
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [5.0, 6.0, 7.0, 8.0],
         [5.0, 6.0, 7.0, 8.0]]

        Alternatively, the range of the operator can be provided directly.
        This requires that the partitions match, i.e. that the cell sizes
        are the same and there is no shift:

        >>> # Same space as in the first example, see above
        >>> large_spc = odl.uniform_discr([-0.5, 0], [1.5, 1], (4, 4))
        >>> resize_op = odl.ResizingOperator(space, large_spc,
        ...                                  pad_mode='periodic')
        >>> print(resize_op(x))
        [[5.0, 6.0, 7.0, 8.0],
         [1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [1.0, 2.0, 3.0, 4.0]]
        """
        if not isinstance(domain, DiscreteLp):
            raise TypeError('`domain` must be a `DiscreteLp` instance, '
                            'got {!r}'.format(domain))

        if not domain.is_uniform:
            raise ValueError('`domain` is not uniformly discretized')

        offset = kwargs.pop('offset', None)
        discr_kwargs = kwargs.pop('discr_kwargs', {})

        if range is None:
            if ran_shp is None:
                raise ValueError('either `range` or `ran_shp` must be '
                                 'given')

            offset = normalized_scalar_param_list(
                offset, domain.ndim, param_conv=safe_int_conv, keep_none=True)

            range = _resize_discr(domain, ran_shp, offset, discr_kwargs)
            self.__offset = tuple(_offset_from_spaces(domain, range))

        elif ran_shp is None:
            if offset is not None:
                raise ValueError('`offset` can only be combined with '
                                 '`ran_shp`')

            if not np.allclose(range.cell_sides, domain.cell_sides):
                raise ValueError(
                    'cell sides of domain and range differ significantly '
                    '(difference {})'
                    ''.format(range.cell_sides - domain.cell_sides))

            self.__offset = _offset_from_spaces(domain, range)

        else:
            raise ValueError('cannot combine `range` with `ran_shape`')

        pad_mode = kwargs.pop('pad_mode', 'constant')
        pad_mode, pad_mode_in = str(pad_mode).lower(), pad_mode
        if pad_mode not in _SUPPORTED_RESIZE_PAD_MODES:
            raise ValueError("`pad_mode` '{}' not understood"
                             "".format(pad_mode_in))

        self.__pad_mode = pad_mode
        # Store constant in a way that ensures safe casting (one-element array)
        self.__pad_const = np.array(kwargs.pop('pad_const', 0),
                                    dtype=range.dtype)

        # padding mode 'constant' with `pad_const != 0` is not linear
        linear = (self.pad_mode != 'constant' or self.pad_const == 0.0)

        super().__init__(domain, range, linear=linear)

    @property
    def offset(self):
        """Number of cells added to/removed from the left."""
        return self.__offset

    @property
    def pad_mode(self):
        """Padding mode used by this operator."""
        return self.__pad_mode

    @property
    def pad_const(self):
        """Constant used by this operator in case of constant padding."""
        return self.__pad_const


class ResizingOperator(ResizingOperatorBase):

    """Operator mapping a discretized function to a new domain.

    This operator is a mapping between uniformly discretized
    `DiscreteLp` spaces with the same `DiscreteLp.cell_sides`,
    but different `DiscreteLp.shape`. The underlying operation is array
    resizing, i.e. no resampling is performed.
    In axes where the domain is enlarged, the new entries are filled
    ("padded") according to a provided parameter ``pad_mode``.

    All resizing operator variants are linear, except constant padding
    with constant != 0.

    See `the online documentation
    <https://odlgroup.github.io/odl/math/resizing_ops.html>`_
    on resizing operators for mathematical details.
    """

    def _call(self, x, out):
        """Implement ``self(x, out)``."""
        # TODO: simplify once context manager is available
        out[:] = resize_array(x.asarray(), self.range.shape,
                              offset=self.offset, pad_mode=self.pad_mode,
                              pad_const=self.pad_const, direction='forward',
                              out=out.asarray())

    def derivative(self, point):
        """Derivative of this operator at ``point``.

        For the particular case of constant padding with non-zero
        constant, the derivative is the corresponding zero-padding
        variant. In all other cases, this operator is linear, i.e.
        the derivative is equal to ``self``.
        """
        if self.pad_mode == 'constant' and self.pad_const != 0:
            return ResizingOperator(
                domain=self.domain, range=self.range, pad_mode='constant',
                pad_const=0.0)
        else:  # operator is linear
            return self

    @property
    def adjoint(self):
        """Adjoint of this operator."""
        if not self.is_linear:
            raise NotImplementedError('this operator is not linear and '
                                      'thus has no adjoint')

        forward_op = self

        class ResizingOperatorAdjoint(ResizingOperatorBase):

            """Adjoint of `ResizingOperator`.

            See `the online documentation
            <https://odlgroup.github.io/odl/math/resizing_ops.html>`_
            on resizing operators for mathematical details.
            """

            def _call(self, x, out):
                """Implement ``self(x, out)``."""
                # TODO: simplify once context manager is available
                out[:] = resize_array(
                    x.asarray(), self.range.shape, offset=self.offset,
                    pad_mode=self.pad_mode, pad_const=0, direction='adjoint',
                    out=out.asarray())

            @property
            def adjoint(self):
                """Adjoint of the adjoint, i.e. the original operator."""
                return forward_op

            @property
            def inverse(self):
                """(Pseudo-)Inverse of this operator.

                Note that in axes where ``self`` extends, the returned operator
                acts as a proper inverse, while in restriction axes, the
                operation is not invertible.
                """
                return ResizingOperatorAdjoint(
                    domain=self.range, range=self.domain,
                    pad_mode=self.pad_mode)

        return ResizingOperatorAdjoint(domain=self.range, range=self.domain,
                                       pad_mode=self.pad_mode)

    @property
    def inverse(self):
        """(Pseudo-)Inverse of this operator.

        Note that in axes where ``self`` extends, the returned operator
        acts as left inverse, while in restriction axes, it is a
        right inverse.
        """
        return ResizingOperator(domain=self.range, range=self.domain,
                                pad_mode=self.pad_mode,
                                pad_const=self.pad_const)


def _offset_from_spaces(dom, ran):
    """Return index offset corresponding to given spaces."""
    diff_l = np.abs(ran.grid.min() - dom.grid.min())
    offset_float = diff_l / dom.cell_sides
    offset = np.around(offset_float).astype(int)
    if not np.allclose(offset, offset_float):
        raise ValueError('range is shifted relative to domain by a '
                         'non-multiple {} of cell_sides'
                         ''.format(offset_float - offset))
    return tuple(offset)


def _resize_discr(discr, newshp, offset, discr_kwargs):
    """Return a space based on ``discr`` and ``newshp``.

    Use the domain of ``discr`` and its partition to create a new
    uniformly discretized space with ``newshp`` as shape. In axes where
    ``offset`` is given, it determines the number of added/removed cells to
    the left. Where ``offset`` is ``None``, the points are distributed
    evenly to left and right. The ``discr_kwargs`` parameter is passed
    to `uniform_discr` for further specification of discretization
    parameters.
    """
    nodes_on_bdry = discr_kwargs.get('nodes_on_bdry', False)
    if np.shape(nodes_on_bdry) == ():
        nodes_on_bdry = ([(bool(nodes_on_bdry), bool(nodes_on_bdry))] *
                         discr.ndim)
    elif discr.ndim == 1 and len(nodes_on_bdry) == 2:
        nodes_on_bdry = [nodes_on_bdry]
    elif len(nodes_on_bdry) != discr.ndim:
        raise ValueError('`nodes_on_bdry` has length {}, expected {}'
                         ''.format(len(nodes_on_bdry), discr.ndim))

    dtype = discr_kwargs.pop('dtype', discr.dtype)
    impl = discr_kwargs.pop('impl', discr.impl)
    exponent = discr_kwargs.pop('exponent', discr.exponent)
    interp = discr_kwargs.pop('interp', discr.interp)
    order = discr_kwargs.pop('order', discr.order)
    weighting = discr_kwargs.pop('weighting', discr.weighting)

    grid_min, grid_max = discr.grid.min(), discr.grid.max()
    cell_size = discr.cell_sides
    new_minpt, new_maxpt = [], []

    for axis, (n_orig, n_new, off, on_bdry) in enumerate(zip(
            discr.shape, newshp, offset, nodes_on_bdry)):

        n_diff = n_new - n_orig
        if off is None:
            num_r = n_diff // 2
            num_l = n_diff - num_r
        else:
            num_r = n_diff - off
            num_l = off

        try:
            on_bdry_l, on_bdry_r = on_bdry
        except TypeError:
            on_bdry_l = on_bdry
            on_bdry_r = on_bdry

        if on_bdry_l:
            new_minpt.append(grid_min[axis] - num_l * cell_size[axis])
        else:
            new_minpt.append(grid_min[axis] - (num_l + 0.5) * cell_size[axis])

        if on_bdry_r:
            new_maxpt.append(grid_max[axis] + num_r * cell_size[axis])
        else:
            new_maxpt.append(grid_max[axis] + (num_r + 0.5) * cell_size[axis])

    return uniform_discr(new_minpt, new_maxpt, newshp, dtype=dtype, impl=impl,
                         exponent=exponent, interp=interp, order=order,
                         weighting=weighting, nodes_on_bdry=nodes_on_bdry)

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
