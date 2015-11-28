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

"""Discrete wavelet transformation on L2 spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, str, super

# External
import numpy as np

# Internal
from odl.discr.lp_discr import DiscreteLp
from odl.operator.operator import Operator

try:
    import pywt
    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False

__all__ = ('DiscreteWaveletTrafo', 'DiscreteWaveletTrafoAdjoint',
           'DiscreteWaveletTrafoInverse', 'WAVELET_AVAILABLE')


_SUPPORTED_IMPL = ('pywt',)


def list_of_coeff_sizes(shape, nscale, wbasis, mode):
    """Construct a size list from given wavelet coefficients.

    Construct a list containing the sizes of the wavelet approximation
    and detail coefficients when wavelet basis, number of scaling levels
    and the shape of the original image are given.
    Related to 2D and 3D multidimensional wavelet transform.

    Parameters
    ----------
    shape : `tuple`
        Number of pixels/voxels in the image. Its lenght must be 2 or 3.
    nscale : `int`
        Number of scales in the multidimensional wavelet
        transform
    wbasis : ``_pywt.Wavelet``
        Describes properties of a selected wavelet basis
    mode : `str`
        Signal extention mode. Possible extension modes are
        'zpd' - zero-padding: signal is extended by adding zero samples
        'cpd' - constant padding: border values are replicated
        'sym' - symmetric padding: signal extension by mirroring samples
        'ppd' - periodic padding: signal is trated as a periodic one
        'sp1' - smooth padding: signal is extended according to the
        first derivatives calculated on the edges (straight line)
        'per' - periodization: like periodic-padding but gives the
        smallest possible number of decomposition coefficients.

    Returns
    -------
    size_list : `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels
        size_list[0] = size of approximation coefficients at the coarsest level
        size_list[1] = size of the detailed coefficients at the coarsest level
        ...
        size_list[n] = size of the detailed coefficients at the finest level
        size_list[n+1] = size of original image
        n = number of scaling levels = nscale

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """
    if len(shape) not in (2, 3):
        raise ValueError('shape must have length 2 or 3, got {}.'
                         ''.format(len(shape)))
    P = np.zeros(shape)
    max_level = pywt.dwt_max_level(shape[0], filter_len=wbasis.dec_len)
    if nscale >= max_level:
        raise ValueError('Too many scaling levels')

    if len(shape) == 2:
        W = pywt.wavedec2(P, wbasis, mode, level=nscale)
    if len(shape) == 3:
        W = wavelet_decomposition3d(P, wbasis, mode, nscale)

    size_list = []
    a = np.shape(W[0])
    size_list.append(a)
    for kk in range(1, nscale+1):
        a = np.shape(W[kk])
        a = a[1:]
        size_list.append(a)

    size_list.append(shape)
    return size_list


def pywt_coeff_to_array2d(coeff, size_list, nscales):
    """Convert a pywt coefficient into a flat array.

    Related to 2D multilevel discrete wavelet transform.

    Parameters
    ----------
    coeff : `list`
        Coefficient are organized in the list in the following way
        [aaN, (adN, daN, ddN), ... (ad1, da1, dd1)]
        The appreviations refer to,

        aa = approx. on 1st dim, approx. on 2nd dim (approximation),

        ad = approx. on 1st dim, detail on 2nd dim (horizontal),

        da = detail on 1st dim, approx. on 2nd dim (vertical),

        dd = detail on 1st dim, detail on 2nd dim (diaginal),

        N = the level of decomposition i.e. number of scales.

    size_list : `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels,

        size_list[0] = size of approximation coefficients at
            the coarsest level, i.e. size of aaN,
        size_list[1] = size of the detailed coefficients at
            the coarsest level, i.e.  size of adN/daN/ddN,
        size_list[N] = size of the detailed coefficients at
            the finest level, i.e. size of ad1/da1/dd1,
        size_list[N+1] = size of original image,
        N = nscales

    nscales : `int`
        Number of scales in the coefficient list

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed and on the chosen wavelet basis.
        If the size of the input image is 2^n x 2^n, the lenght of the
        wavelet coefficient array is the same.

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """
    # TODO: outsource to an own helper?
    size_flatCoeff = np.prod(size_list[0])
    for kk in range(1, nscales+1):
        size_flatCoeff = size_flatCoeff + 3*np.prod(size_list[kk])

    flat_coeff = np.zeros((size_flatCoeff))
    aa = coeff[0]
    aa = aa.ravel()
    stop = np.prod(size_list[0])
    flat_coeff[0:stop] = aa
    start = stop
    for kk in range(1, nscales+1):
        (ad, da, dd) = coeff[kk]
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = ad.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = da.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = dd.ravel()
        start = stop
    return flat_coeff


def array_to_pywt_coeff2d(coeff, size_list, nscales):
    """Convert a flat array into a pywt coefficient list.
    For multilevel 2D discrete wavelet transform

    Parameters
    ----------
    coeff :  :class:`DiscreteLp.Vector`
        A flat coefficient vector containing the approximation,
        horizontal detail, vertical detail and diagonal detail
        coefficients in the following order,
        [aaN, adN, daN, ddN, ... ad1, da1, dd1],

        The appreviations refer to,

        aa = approx. on 1st dim, approx. on 2nd dim (approximation)

        ad = approx. on 1st dim, detail on 2nd dim (horizontal)

        da = detail on 1st dim, approx. on 2nd dim (vertical)

        dd = detail on 1st dim, detail on 2nd dim (diagonal)

    size_list : `list`
       A list of coefficient sizes such that,

       size_list[0] = size of approximation coefficients at the coarsest level,

       size_list[1] = size of the detailed coefficients at the coarsest level,

       size_list[N] = size of the detailed coefficients at the finest level,

       size_list[N+1] = size of original image,

       N = nscales

    nscales : `int`
        Number of scales in the coefficient list

    Returns
    -------
    :attr:`coeff_list` : `list`
        A list of coefficient organized in the following way
        [aaN, (adN, daN, ddN), ... (ad1, da1, dd1)],

        The appreviations refer to

        aa = approx. on 1st dim, approx. on 2nd dim

        ad = approx. on 1st dim, detail on 2nd dim (horizontal)

        da = detail on 1st dim, approx. on 2nd dim (vertical)

        dd = detail on 1st dim, detail on 2nd dim (diaginal)

        N = the level of decomposition/number of scales.

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """
    size1 = np.prod(size_list[0])
    aa_flat = coeff[0:size1]
    aa = np.asarray(aa_flat).reshape(size_list[0])
    kk = 1
    coeff_list = []
    coeff_list.append(aa)

    while kk <= nscales:
        detail_shape = size_list[kk]
        size2 = np.prod(size_list[kk])
        ad_flat = coeff[size1:size1+size2]
        ad = np.asarray(ad_flat).reshape(detail_shape)

        da_flat = coeff[size1+size2:size1+2*size2]
        da = np.asarray(da_flat).reshape(detail_shape)

        dd_flat = coeff[size1+2*size2:size1 + 3*size1]
        dd = np.asarray(dd_flat).reshape(detail_shape)

        details = (ad, da, dd)
        coeff_list.append(details)

        kk = kk + 1
        size1 = size1 + 3*size2

    return coeff_list


def pywt_coeff_to_array3d(coeff, size_list, nscales):
    """Convert a pywt coefficient into a flat array.

    Related to 3D multilevel discrete wavelet transform.

    Parameters
    ----------
    coeff : `list`
        Coefficient are organized in the list in the following way
        [aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)].
        The appreviations refer to

        aaa = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        aad = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ada = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        add = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        daa = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        dad = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        dda = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ddd = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        N =  the number of scaling levels

    size_list : `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels

        size_list[0] = size of approximation coefficients at
            the coarsest level,
        size_list[1] = size of the detailed coefficients at
            the coarsest level,
        size_list[N] = size of the detailed coefficients at
            the finest level,
        size_list[N+1] = size of original image,
        N = number of scales

    nscales : `int`
        Number of scales in the coefficient list

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed and on the chosen wavelet basis.
        If the size of the input image is 2^n x 2^n, the lenght of the
        wavelet coefficient array is the same.

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """
    size_flatCoeff = np.prod(size_list[0])
    for kk in range(1, nscales+1):
        size_flatCoeff = size_flatCoeff + 7*np.prod(size_list[kk])

    flat_coeff = np.zeros((size_flatCoeff))
    approx = coeff[0]
    approx = approx.ravel()
    stop = np.prod(size_list[0])
    flat_coeff[0:stop] = approx
    start = stop
    for kk in range(1, nscales+1):
        (aad, ada, add, daa, dad, dda, ddd) = coeff[kk]
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = aad.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = ada.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = add.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = daa.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = dad.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = dda.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = ddd.ravel()
        start = stop

    return flat_coeff


def array_to_pywt_coeff3d(coeff, size_list, nscales):
    """Convert a flat array into a pywt coefficient list.

    For multilevel 3D discrete wavelet transform

    Parameters
    ----------
    coeff :  :class:`DiscreteLp.Vector`
        A flat coefficient vector containing the approximation,
        and detail coefficients in the following order
        [aaaN, aadN, adaN, addN, daaN, dadN, ddaN, dddN, ...
        aad1, ada1, add1, daa1, dad1, dda1, ddd1]

    size_list : list
       A list of coefficient sizes such that,

       size_list[0] = size of approximation coefficients at the coarsest level,

       size_list[1] = size of the detailed coefficients at the coarsest level,

       size_list[N] = size of the detailed coefficients at the finest level,

       size_list[N+1] = size of original image,

       N = nscales

    nscales : int
        Number of scales in the coefficient list

    Returns
    -------
    :attr:`coeff_list` : `list`
        A list of coefficient organized in the following way
        [aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)].
        The appreviations refer to,

        aaa = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        aad = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ada = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        add = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        daa = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        dad = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        dda = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ddd = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        N = the number of scaling levels

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """

    size1 = np.prod(size_list[0])
    approx_flat = coeff[0:size1]
    approx = np.asarray(approx_flat).reshape(size_list[0])
    kk = 1
    coeff_list = []
    coeff_list.append(approx)

    while kk <= nscales:
        detail_shape = size_list[kk]
        size2 = np.prod(size_list[kk])
        aad_flat = coeff[size1:size1+size2]
        aad = np.asarray(aad_flat).reshape(detail_shape)

        ada_flat = coeff[size1+size2:size1+2*size2]
        ada = np.asarray(ada_flat).reshape(detail_shape)

        add_flat = coeff[size1+2*size2:size1 + 3*size1]
        add = np.asarray(add_flat).reshape(detail_shape)

        daa_flat = coeff[size1+3*size2:size1 + 4*size1]
        daa = np.asarray(daa_flat).reshape(detail_shape)

        dad_flat = coeff[size1+4*size2:size1 + 5*size1]
        dad = np.asarray(dad_flat).reshape(detail_shape)

        dda_flat = coeff[size1+5*size2:size1 + 6*size1]
        dda = np.asarray(dda_flat).reshape(detail_shape)

        ddd_flat = coeff[size1+6*size2:size1 + 7*size1]
        ddd = np.asarray(ddd_flat).reshape(detail_shape)

        details = (aad, ada, add, daa, dad, dda, ddd)
        coeff_list.append(details)

        kk = kk + 1
        size1 = size1 + 7*size2

    return coeff_list


def wavelet_decomposition3d(x, wbasis, mode, nscales):
    """Discrete 3D multiresolution wavelet decomposition.

    Compute the discrete 3D multiresolution wavelet decomposition
    at the given level (nscales) for a given 3D image.
    Utilizes a PyWavelet function ``pywt.dwtn``.

    Parameters
    ----------
        x : `DiscreteLp.Vector`
        wbasis:  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basis
        mode : `str`
            Signal extention mode. For possible extensions see
            ``pywt.MODES.modes``
        nscales : `int`
            Number of scales in the coefficient list

    Returns
    -------
        :attr:`coeff_list` : `list`
        A list of coefficient organized in the following way
         [aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
         (aad1, ada1, add1, daa1, dad1, dda1, ddd1)] .
         The appreviations refer to,

         aaa = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

         aad = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

         ada = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

         add = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

         daa = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

         dad = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

         dda = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

         ddd = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

         N = the number of scaling levels

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """

    wcoeffs = pywt.dwtn(x, wbasis, mode)
    aaa = wcoeffs['aaa']
    aad = wcoeffs['aad']
    ada = wcoeffs['ada']
    add = wcoeffs['add']
    daa = wcoeffs['daa']
    dad = wcoeffs['dad']
    dda = wcoeffs['dda']
    ddd = wcoeffs['ddd']

    details = (aad, ada, add, daa, dad, dda, ddd)
    coeff_list = []
    coeff_list.append(details)

    for kk in range(1, nscales):
        wcoeffs = pywt.dwtn(aaa, wbasis, mode)
        aaa = wcoeffs['aaa']
        aad = wcoeffs['aad']
        ada = wcoeffs['ada']
        add = wcoeffs['add']
        daa = wcoeffs['daa']
        dad = wcoeffs['dad']
        dda = wcoeffs['dda']
        ddd = wcoeffs['ddd']
        details = (aad, ada, add, daa, dad, dda, ddd)
        coeff_list.append(details)

    coeff_list.append(aaa)
    coeff_list.reverse()

    return coeff_list


def wavelet_reconstruction3d(coeff_list, wbasis, mode, nscales):
    """Discrete 3D multiresolution wavelet reconstruction

    Compute a discrete 3D multiresolution wavelet reconstruction
    from a given wavelet coefficient list.
    Utilizes a PyWavelet function ``pywt.dwtn``

    Parameters
    ----------
        coeff_list: : `list`
            A list of wavelet approximation and detail coefficients
            organized in the following way
            [caaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
            (aad1, ada1, add1, daa1, dad1, dda1, ddd1)].
            The appreviations refer to,

            aaa = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

            aad = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

            ada = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

            add = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

            daa = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

            dad = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

            dda = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

            ddd = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

            N = the number of scaling levels
        wbasis :  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basis
        mode : `str`
            Signal extention mode. For possible extensions see
            ``pywt.MODES.modes``
        nscales : `int`
            Number of scales in the coefficient list

    Returns
    -------
        x : `numpy.ndarray`.
        A wavalet reconstruction.

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """
    aaa = coeff_list[0]
    k = 1
    (aad, ada, add, daa, dad, dda, ddd) = coeff_list[k]
    coeff_dict = {'aaa': aaa, 'aad': aad, 'ada': ada, 'add': add,
                  'daa': daa, 'dad': dad, 'dda': dda, 'ddd': ddd}
    for k in range(2, nscales+1):
        aaa = pywt.idwtn(coeff_dict, wbasis, mode)
        (aad, ada, add, daa, dad, dda, ddd) = coeff_list[k]
        coeff_dict = {'aaa': aaa, 'aad': aad, 'ada': ada, 'add': add,
                      'daa': daa, 'dad': dad, 'dda': dda, 'ddd': ddd}

    x = pywt.idwtn(coeff_dict, wbasis, mode)
    return x


class DiscreteWaveletTrafo(Operator):

    """Discrete wavelet trafo between discrete L2 spaces."""

    def __init__(self, dom, nscales, wbasis, mode):
        """Initialize a new instance.

        Parameters
        ----------
        dom : :class:`~odl.discr.lp_discr.DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent :math:`p` of the discrete :math:`L^p`
            space must be equal to 2.0.
        nscales : `int`
            Number of scales in the coefficient list
        wbasis :  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basis
        mode : `str`
            Signal extention mode. For possible extensions see
            ``pywt.MODES.modes``

        See also
        --------
        pywt : http://www.pybytes.com/pywavelets/
        """
        self.nscales = int(nscales)
        self.wbasis = wbasis
        self.mode = str(mode).lower()

        max_level = pywt.dwt_max_level(dom.grid.shape[0],
                                       filter_len=self.wbasis.dec_len)
        # TODO: maybe the error message could tell how to calculate the
        # max number of levels
        if self.nscales >= max_level:
            raise ValueError('Cannot use more than {} scaling levels, '
                             'got {}.'.format(max_level, self.nscales))

        self.size_list = list_of_coeff_sizes(dom.grid.shape, self.nscales,
                                             self.wbasis, self.mode)

        ran_size = np.prod(self.size_list[0])

        if len(dom.grid.shape) == 2:
            ran_size += sum(3 * np.prod(shape) for shape in
                            self.size_list[1:-1])
        if len(dom.grid.shape) == 3:
            ran_size += sum(7 * np.prod(shape) for shape in
                            self.size_list[1:-1])

        # TODO: Maybe allow other ranges like Besov spaces (yet to be crated)
        ran = dom.dspace_type(ran_size, dtype=dom.dtype)
        super().__init__(dom, ran)

        if not isinstance(dom, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))

        if dom.exponent != 2.0:
            raise ValueError('domain Lp exponent is {} instead of 2.0.'
                             ''.format(dom.exponent))
#        if not np.all(dom.grid.stride == 1):
#            raise NotImplementedError('non-uniform grid cell sizes not yet '
#                                      'supported.')

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, x):
        """Compute the discrete 2D and 3D multiresolution wavelet transform
        Parameters
        ----------
        x : `DiscreteLp.Vector`

        Returns
        -------
        arr : `numpy.ndarray`
            Flattened and concatenated coefficient array
            The length of the array depends on the size of input image to
            be transformed and on the chosen wavelet basis.
        """
        if len(x.shape) == 2:
            coeff_list = pywt.wavedec2(x, self.wbasis, self.mode, self.nscales)
            coeff_arr = pywt_coeff_to_array2d(coeff_list, self.size_list,
                                              self.nscales)
            return self.range.element(coeff_arr)

        if len(x.shape) == 3:
            coeff_dict = wavelet_decomposition3d(x, self.wbasis, self.mode,
                                                 self.nscales)
            coeff_arr = pywt_coeff_to_array3d(coeff_dict, self.size_list,
                                              self.nscales)

            return self.range.element(coeff_arr)

    @property
    def adjoint(self):
        """The adjoint wavelet transform."""
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return None

    @property
    def inverse(self):
        """The inverse wavelet transform."""

        return DiscreteWaveletTrafoInverse(ran=self.domain,
                                           nscales=self.nscales,
                                           wbasis=self.wbasis, mode=self.mode)


class DiscreteWaveletTrafoAdjoint(Operator):
    pass


class DiscreteWaveletTrafoInverse(Operator):

    """Discrete inverse wavelet trafo between discrete L2 spaces."""

    def __init__(self, ran, nscales, wbasis, mode):
        """Initialize a new instance.

        Parameters
        ----------
        ran : `odl.discr.lp_discr.DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent `p` of the discrete :math:`L^p`
            space must be equal to 2.0.
        nscales : `int`
            Number of scales in the coefficient list
        wbasis :  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basis
        mode : `str`
            Signal extention mode. For possible extensions see
            ``pywt.MODES.modes``
        See also
        --------
        pywt : http://www.pybytes.com/pywavelets/
        """
        self.nscales = int(nscales)
        self.wbasis = wbasis
        self.mode = str(mode).lower()

        max_level = pywt.dwt_max_level(ran.grid.shape[0],
                                       filter_len=self.wbasis.dec_len)
        # TODO: maybe the error message could tell how to calculate the
        # max number of levels
        if self.nscales >= max_level:
            raise ValueError('Cannot use more than {} scaling levels, '
                             'got {}.'.format(max_level, self.nscales))

        self.size_list = list_of_coeff_sizes(ran.grid.shape, self.nscales,
                                             self.wbasis, self.mode)

        dom_size = np.prod(self.size_list[0])
        if len(ran.grid.shape) == 2:
            dom_size += sum(3 * np.prod(shape) for shape in
                            self.size_list[1:-1])
        if len(ran.grid.shape) == 3:
            dom_size += sum(7 * np.prod(shape) for shape in
                            self.size_list[1:-1])

        # TODO: Maybe allow other ranges like Besov spaces (yet to be created)
        dom = ran.dspace_type(dom_size, dtype=ran.dtype)
        super().__init__(dom, ran)

        if not isinstance(ran, DiscreteLp):
            raise TypeError('range {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))

        if ran.exponent != 2.0:
            raise ValueError('range Lp exponent is {} instead of 2.0.'
                             ''.format(ran.exponent))
#        if not np.all(dom.grid.stride == 1):
#            raise NotImplementedError('non-uniform grid cell sizes not yet '
#                                      'supported.')

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, coeff):
        """Compute the discrete 2D and 3D inverse
        multiresolution wavelet transform

        Parameters
        ----------
        coeff : class:`DiscreteLp.Vector`

        Returns
        -------
        arr : `DiscreteLp.Vector`

        """
        if len(self.range.grid.shape) == 2:
            coeff_list = array_to_pywt_coeff2d(coeff, self.size_list,
                                               self.nscales)
            x = pywt.waverec2(coeff_list, self.wbasis, self.mode)
            return self.range.element(x)
        if len(self.range.grid.shape) == 3:
            coeff_dict = array_to_pywt_coeff3d(coeff, self.size_list,
                                               self.nscales)
            x = wavelet_reconstruction3d(coeff_dict, self.wbasis, self.mode,
                                         self.nscales)
            return self.range.element(x)

    @property
    def adjoint(self):
        """The adjoint wavelet transform."""
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return None

    @property
    def inverse(self):
        """The inverse wavelet transform."""
        return DiscreteWaveletTrafo(dom=self.range, nscales=self.nscales,
                                    wbasis=self.wbasis, mode=self.mode)
