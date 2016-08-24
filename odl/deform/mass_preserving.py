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

"""Operators and functions for general and mass-preserving deformation."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.discr import DiscreteLp, DiscreteLpVector
from odl.operator.operator import Operator
from odl.space import ProductSpaceVector, ProductSpace


__all__ = ('GeneralDeformFixedTempl', 'GeneralDeformFixedDeform',
           'MassPresvDeformFixedTempl', 'MassPresvDeformFixedDeform')


def _general_deform(template, deformation, out=None):
    """Deform a template with a general deformation field.

    The function maps a given template ``I`` and a given deformation
    field ``phi`` to the new function ``x --> I(phi(x))``.

    Parameters
    ----------
    template : `DiscreteLpVector`
        Template to be deformed by a deformation field.
    deformation : element of power space of `DiscreteLp`
        The deformation field used to deform the template.
    out : `numpy.ndarray`, optional
        Array to which the function values of the deformed template
        are written. It must have the same shape as ``template`` and
        a data type compatible with ``template.dtype``.

    Returns
    -------
    deformed_template : `numpy.ndarray`
        Function values of the deformed template. If ``out`` was given,
        the returned object is a reference to it.

    Examples
    --------
    Create a simple 1D template and a deformation field. Where the deformation
    is identity, the output value is the same as the input value.
    In the 4-th point, the value is taken from 0.2 (one cell) to
    the left, i.e. 1.0.

    >>> import odl
    >>> space = odl.uniform_discr(0, 1, 5, interp='nearest')
    >>> deform_field_space = space.vector_field_space
    >>> template = space.element([0, 0, 1, 0, 0])
    >>> deform_field = deform_field_space.element([[0.1, 0.3, 0.5, 0.5, 0.9]])
    >>> _general_deform(template, deform_field)
    array([ 0.,  0.,  1.,  1.,  0.])

    The result depends on the chosen interpolation. With 'linear'
    interpolation and an offset equal to half the distance between two
    points, 0.1, one gets the mean of the values.

    >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
    >>> deform_field_space = space.vector_field_space
    >>> template = space.element([0, 0, 1, 0, 0])
    >>> deform_field = deform_field_space.element([[0.1, 0.3, 0.5, 0.6, 0.9]])
    >>> _general_deform(template, deform_field)
    array([ 0. ,  0. ,  1. ,  0.5,  0. ])
    """
    image_pts = template.space.points()
    for i, vi in enumerate(deformation):
        image_pts[:, i] = vi.ntuple.asarray()
    return template.interpolation(image_pts.T, out=out, bounds_check=False)


def _mass_presv_deform(template, mass_presv_dfield, out=None):
    """Deform a template with a mass-preserving deformation field.

    The function maps a given template ``I``, a mass-preserving deformation
    field, i.e., a given deformation field ``phi`` and its Jacobian
    determinant to the new function ``x --> |Dphi(x)|I(phi(x))``.

    Parameters
    ----------
    template : `DiscreteLpVector`
        Template to be deformed by a deformation field.
    mass_presv_dfield : element of power space of power space and `DiscreteLp`
        The mass-preserving deformation field used to deform the template.
        It contains the deformation, i.e., an element of power space
        of `DiscreteLp`, and the Jacobian determinant of the deformation,
        i.e., a `DiscreteLpVector`.
    out : `numpy.ndarray`, optional
        Array to which the function values of the deformed template
        are written. It must have the same shape as ``template`` and
        a data type compatible with ``template.dtype``.

    Returns
    -------
    deformed_template : `numpy.ndarray`
        Function values of the deformed template. If ``out`` was given,
        the returned object is a reference to it.

    Examples
    --------
    Create a simple 1D template ``I`` over ``[0, 1]``, a deffeomorphic
    deformation field and its Jacobian determinant.
    If the deffeomorphism ``phi`` is ``sin(x*pi/2)``, its Jacobian determinant
    should be ``cos(x*pi/2)*pi/2`` and the output value is
    ``cos(x*pi/2)*pi/2*I(sin(x*pi/2))``. A mass-preserving deformation
    is obtained, namely, the integration of ``I(x)`` over ``[0, 1]``
    equals to the integration of ``|Dphi(x)|I(phi(x))`` over ``[0, 1]``.


    >>> import odl
    >>> import numpy as np
    >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
    >>> vspace = space.vector_field_space
    >>> pspace = odl.ProductSpace(vspace, space)
    >>> template = space.element([0, 0, 1, 0, 0])
    >>> deform_field = vspace.element([np.sin(space.points() * np.pi / 2.0)])
    >>> det = space.element(np.cos(space.points() * np.pi / 2.0) * np.pi / 2.0)
    >>> mass_presv_field = pspace.element()
    >>> mass_presv_field[0] = deform_field
    >>> mass_presv_field[1] = det
    >>> _mass_presv_deform(template, mass_presv_field)
    uniform_discr(0.0, 1.0, 5, interp=u'linear').element([0.0,
    1.0776176446789727, 0.0, 0.0, 0.0])
    """
    general_deform_template = _general_deform(template, mass_presv_dfield[0])
    return mass_presv_dfield[1] * general_deform_template


class GeneralDeformFixedTempl(Operator):

    """Deformation operator with fixed template acting on deformation fields.

    The operator has a fixed template ``I`` and maps a deformation
    field ``phi`` to the new function ``x --> I(phi(x))``.

    See Also
    --------
    GeneralDeformFixedDeform : Deformation with a fixed deformation.

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d`, we take :math:`X = L^p(\Omega)`
    to be the template space, i.e. :math:`I \\in X`. Then the deformation field
    space is identified with :math:`G := X^d`. Hence the deformation operator
    with fixed template maps :math:`G` into :math:`X`:

    .. math::

        W_I : G \\to X, \quad W_I(\phi) := I(\phi(\cdot)),

    i.e., :math:`W_I(\phi)(x) = I(\phi(x))`.
    """

    def __init__(self, template):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template that is to be deformed.

        Examples
        --------
        Create a simple 1D template to initialize the operator and
        apply it to a deformation field. Where the deformation is identity,
        the output value is the same as the input value.
        In the 4-th point, the value is taken from 0.2 (one cell) to the
        left, i.e. 1.0.

        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5, interp='nearest')
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = GeneralDeformFixedTempl(template)
        >>> deform_field_space = space.vector_field_space
        >>> deform = deform_field_space.element([[0.1, 0.3, 0.5, 0.5, 0.9]])
        >>> print(op(deform))
        [0.0, 0.0, 1.0, 1.0, 0.0]

        The result depends on the chosen interpolation. With 'linear'
        interpolation and an offset equal to half the distance between two
        points, 0.1, one gets the mean of the values.

        >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = GeneralDeformFixedTempl(template)
        >>> deform_field_space = space.vector_field_space
        >>> deform = deform_field_space.element([[0.1, 0.3, 0.5, 0.6, 0.9]])
        >>> print(op(deform))
        [0.0, 0.0, 1.0, 0.5, 0.0]
        """
        if not isinstance(template, DiscreteLpVector):
            raise TypeError('`template` must be a `DiscreteLpVector`'
                            'instance, got {!r}'.format(template))

        self.__template = template
        super().__init__(domain=self.template.space.vector_field_space,
                         range=self.template.space, linear=False)

    @property
    def template(self):
        """Fixed template of this deformation operator."""
        return self.__template

    def _call(self, deformation, out=None):
        """Implementation of ``self(deformation[, out])``."""
        return _general_deform(self.template, deformation, out)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.template)


class GeneralDeformFixedDeform(Operator):

    """Deformation operator with fixed deformation acting on templates.

    The operator has a fixed deformation field ``phi`` and maps a template
    ``I`` to the new function ``x --> I(phi(x))``.

    See Also
    --------
    GeneralDeformFixedTempl : Deformation with a fixed template.

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d`, we take :math:`G := X^d`
    to be the space of deformation fields, where :math:`X = L^p(\Omega)`
    is the template space. Hence the deformation operator with the fixed
    deformation field :math:`\phi \\in G` maps :math:`X` into :math:`X`:

    .. math::

        W_{\phi} : X \\to X, \quad W_{\phi}(I) := I(\phi(\cdot)),

    i.e., :math:`W_{\phi}(I)(x) = I(\phi(x))`.
    """

    def __init__(self, deformation, templ_space=None):
        """Initialize a new instance.

        Parameters
        ----------
        deformation : element of a power space of `DiscreteLp`
            Fixed deformation field used in the deformation.
        templ_space : `DiscreteLp`, optional
            Template space on which this operator is applied, i.e. the
            operator domain and range. It must fulfill
            ``domain.vector_field_space == deformation.space``, so this
            option is useful mainly for support of complex spaces.
            Default: ``deformation.space[0]``

        Examples
        --------
        Create a simple 1D deformation field to initialize the operator
        and apply it to a template. Where the deformation
        is identity, the output value is the same as the input value.
        In the 4-th point, the value is taken from 0.2 (one cell) to
        the left, i.e. 1.0.

        >>> import odl
        >>> space = odl.uniform_discr(0, 1, 5, interp='nearest')
        >>> deform_field_space = space.vector_field_space
        >>> deform = deform_field_space.element([[0.1, 0.3, 0.5, 0.5, 0.9]])
        >>> op = GeneralDeformFixedDeform(deform)
        >>> template = [0, 0, 1, 0, 0]
        >>> print(op(template))
        [0.0, 0.0, 1.0, 1.0, 0.0]

        The result depends on the chosen interpolation. With 'linear'
        interpolation and an offset equal to half the distance between two
        points, 0.1, one gets the mean of the values.

        >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
        >>> deform_field_space = space.vector_field_space
        >>> deform = deform_field_space.element([[0.1, 0.3, 0.5, 0.6, 0.9]])
        >>> op = GeneralDeformFixedDeform(deform)
        >>> template = [0, 0, 1, 0, 0]
        >>> print(op(template))
        [0.0, 0.0, 1.0, 0.5, 0.0]
        """
        if not isinstance(deformation, ProductSpaceVector):
            raise TypeError('`deformation must be a `ProductSpaceVector` '
                            'instance, got {!r}'.format(deformation))
        if not deformation.space.is_power_space:
            raise TypeError('`deformation.space` must be a power space, '
                            'got {!r}'.format(deformation.space))
        if not isinstance(deformation.space[0], DiscreteLp):
            raise TypeError('`deformation.space[0]` must be a `DiscreteLp` '
                            'instance, got {!r}'.format(deformation.space[0]))

        if templ_space is None:
            templ_space = deformation.space[0]
        else:
            if not isinstance(templ_space, DiscreteLp):
                raise TypeError('`templ_space` must be a `DiscreteLp` '
                                'instance, got {!r}'.format(templ_space))
            if templ_space.vector_field_space != deformation.space:
                raise ValueError('`templ_space.vector_field_space` not equal '
                                 'to `deformation.space` ({} != {})'
                                 ''.format(templ_space.vector_field_space,
                                           deformation.space))

        self.__deformation = deformation
        super().__init__(domain=templ_space, range=templ_space, linear=True)

    @property
    def deformation(self):
        """Fixed deformation field of this deformation operator."""
        return self.__deformation

    def _call(self, template, out=None):
        """Implementation of ``self(template[, out])``."""
        return _general_deform(template, self.deformation, out)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.domain == self.deformation.space[0]:
            domain_repr = ''
        else:
            domain_repr = ', domain={!r}'.format(self.domain)

        return '{}({!r}{})'.format(self.__class__.__name__,
                                   self.deformation,
                                   domain_repr)


class MassPresvDeformFixedTempl(Operator):

    """Mass-preserving deformation operator with fixed template.

    The operator has a fixed template ``I`` and maps a mass-preserving
    deformation field, i.e., a deformation field ``phi`` and its Jacobian
    determinant to the new function ``x --> |Dphi(x)|I(phi(x))``.


    See Also
    --------
    MassPresvDeformFixedDeform : Mass-preserving deformation with fixed
    mass-preserving deformation field.

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d`, we take :math:`X = L^p(\Omega)`
    to be the template space, i.e. :math:`I \\in X`. Then the deformation field
    space is identified with :math:`G := X^d`. Hence the deformation operator
    with fixed template maps :math:`G` into :math:`X`:

    .. math::

        W_I : G \\to X, \quad W_I(\phi) := |D\phi(\cdot)|I(\phi(\cdot)),

    i.e., :math:`W_I(\phi)(x) = |D\phi(x)|I(\phi(x))`.
    """

    def __init__(self, template):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template that is to be deformed.

        Examples
        --------
        Create a simple 1D template ``I`` over ``[0, 1]`` to initialize the
        operator and apply it to a mass-preserving deformation field, i.e.,
        a deffeomorphic deformation field and its Jacobian determinant.
        If the deffeomorphism ``phi`` is ``sin(x*pi/2)``, its Jacobian
        determinant should be ``cos(x*pi/2)*pi/2`` and the output value is
        ``cos(x*pi/2)*pi/2*I(sin(x*pi/2))``. A mass-preserving deformation
        is obtained, namely, the integration of ``I(x)`` over ``[0, 1]``
        equals to the integration of ``|Dphi(x)|I(phi(x))`` over ``[0, 1]``.

        >>> import odl
        >>> import numpy as np
        >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
        >>> vspace = space.vector_field_space
        >>> pspace = ProductSpace(vspace, space)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op = MassPresvDeformFixedTempl(template)
        >>> img_pts = space.points()
        >>> deform_field = vspace.element([np.sin(img_pts * np.pi / 2.0)])
        >>> det = space.element(np.cos(img_pts * np.pi / 2.0) * np.pi / 2.0)
        >>> mass_presv_deform = pspace.element()
        >>> mass_presv_deform[0] = deform_field
        >>> mass_presv_deform[1] = det
        >>> op(mass_presv_deform)
        uniform_discr(0.0, 1.0, 5, interp=u'linear').element([0.0,
        1.0776176446789727, 0.0, 0.0, 0.0])
        """
        if not isinstance(template, DiscreteLpVector):
            raise TypeError('`template` must be a `DiscreteLpVector`'
                            'instance, got {!r}'.format(template))

        self.__template = template

        pspace = ProductSpace(self.template.space.vector_field_space,
                              self.template.space)

        super().__init__(domain=pspace,
                         range=self.template.space, linear=False)

    @property
    def template(self):
        """Fixed template of this deformation operator."""
        return self.__template

    def _call(self, mass_presv_dfield, out=None):
        """Implementation of ``self(deformation[, out])``."""
        return _mass_presv_deform(self.template, mass_presv_dfield)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.template)


class MassPresvDeformFixedDeform(Operator):

    """Mass-preserving deformation operator with fixed deformation.

    The operator has a fixed mass-preserving deformation field, i.e.,
    a deformation field ``phi`` and its Jacobean determinant ``|Dphi|``,
    and maps a template ``I`` to the new function ``x --> |Dphi(x)|I(phi(x))``.


    See Also
    --------
    MassPresvDeformFixedTempl : Mass-preserving deformation with fixed
    template.

    Notes
    -----
    For :math:`\Omega \subset \mathbb{R}^d`, we take :math:`G := X^d`
    to be the space of deformation fields, where :math:`X = L^p(\Omega)`
    is the template space. Hence the deformation operator with the fixed
    deformation field :math:`\phi \\in G` maps :math:`X` into :math:`X`:

    .. math::

        W_{\phi} : X \\to X, \quad W_{\phi}(I) := |D\phi(\cdot)|I(\phi(\cdot)),

    i.e., :math:`W_{\phi}(I)(x) = |D\phi(x)|I(\phi(x))`.
    """

    def __init__(self, mass_presv_deform, templ_space=None):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template that is to be deformed.

        Examples
        --------
        Create a simple 1D template ``I`` over ``[0, 1]`` to initialize the
        operator and apply it to a mass-preserving deformation field, i.e.,
        a deffeomorphic deformation field and its Jacobian determinant.
        If the deffeomorphism ``phi`` is ``sin(x*pi/2)``, its Jacobian
        determinant should be ``cos(x*pi/2)*pi/2`` and the output value is
        ``cos(x*pi/2)*pi/2*I(sin(x*pi/2))``. A mass-preserving deformation
        is obtained, namely, the integration of ``I(x)`` over ``[0, 1]``
        equals to the integration of ``|Dphi(x)|I(phi(x))`` over ``[0, 1]``.

        >>> import odl
        >>> import numpy as np
        >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
        >>> vspace = space.vector_field_space
        >>> pspace = ProductSpace(vspace, space)
        >>> img_pts = space.points()
        >>> deform_field = vspace.element([np.sin(img_pts * np.pi / 2.0)])
        >>> det = space.element(np.cos(img_pts * np.pi / 2.0) * np.pi / 2.0)
        >>> mass_presv_deform = pspace.element()
        >>> mass_presv_deform[0] = deform_field
        >>> mass_presv_deform[1] = det
        >>> op = MassPresvDeformFixedDeform(mass_presv_deform)
        >>> template = space.element([0, 0, 1, 0, 0])
        >>> op(template)
        uniform_discr(0.0, 1.0, 5, interp=u'linear').element([0.0,
        1.0776176446789727, 0.0, 0.0, 0.0])
        """
        if not isinstance(mass_presv_deform[0], ProductSpaceVector):
            raise TypeError('`mass_presv_deform[0] must be '
                            'a `ProductSpaceVector` instance, '
                            'got {!r}'.format(mass_presv_deform[0]))
        if not mass_presv_deform.space[0].is_power_space:
            raise TypeError('`mass_presv_deform.space[0]` must be '
                            'a power space, '
                            'got {!r}'.format(mass_presv_deform.space[0]))
        if not isinstance(mass_presv_deform.space[1], DiscreteLp):
            raise TypeError('`mass_presv_deform.space[1]` must be '
                            'a `DiscreteLp` instance, '
                            'got {!r}'.format(mass_presv_deform.space[1]))

        if templ_space is None:
            templ_space = mass_presv_deform.space[1]
        else:
            if not isinstance(templ_space, DiscreteLp):
                raise TypeError('`templ_space` must be a `DiscreteLp` '
                                'instance, got {!r}'.format(templ_space))
            if templ_space.vector_field_space != mass_presv_deform.space[0]:
                raise ValueError('`templ_space.vector_field_space` not equal '
                                 'to `mass_presv_deform.space[0]` ({} != {})'
                                 ''.format(templ_space.vector_field_space,
                                           mass_presv_deform.space[0]))

        self.__mass_presv_dfield = mass_presv_deform
        super().__init__(domain=templ_space, range=templ_space, linear=True)

    @property
    def mass_presv_deformation(self):
        """Fixed deformation field of this deformation operator."""
        return self.__mass_presv_dfield

    def _call(self, template, out=None):
        """Implementation of ``self(template[, out])``."""
        return _mass_presv_deform(template, self.mass_presv_deformation)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.domain == self.mass_presv_deformation.space[1]:
            domain_repr = ''
        else:
            domain_repr = ', domain={!r}'.format(self.domain)

        return '{}({!r}{})'.format(self.__class__.__name__,
                                   self.mass_presv_deformation,
                                   domain_repr)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
