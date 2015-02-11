# -*- coding: utf-8 -*-
"""
fourier.py -- uniform and non-uniform Fourier transforms

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

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import object
from builtins import super
from future import standard_library
standard_library.install_aliases()
from builtins import range

import numpy as np
from numpy import pi
try:
    import numexpr as ne
except ImportError:
    ne = None
from functools import partial

from RL.datamodel import ugrid as ug
from RL.datamodel import gfunc as gf
from RL.utility.utility import errfmt, InputValidationError, SQRT_2PI


class FourierTransform(object):
    """Basic Fourier transform class. Will select the subclass depending
    based on the arguments given for initialization.
    TODO: write some more
    """
    def __new__(cls, *args, **kwargs):
        try:
            in_arg = args[0]
        except IndexError:
            in_arg = kwargs.get('in', None)

        try:
            out_arg = args[1]
        except IndexError:
            out_arg = kwargs.get('out', None)

        if in_arg is None:
            raise ValueError("Unable to determine input type.")

        if isinstance(in_arg, gf.Gfunc):
            if isinstance(out_arg, gf.Gfunc) or out_arg is None:
                cls = FourierTransformUniUni
            elif isinstance(out_arg, np.ndarray):
                cls = FourierTransformUniNonuni
            else:
                raise ValueError(errfmt("""\
                No transform defined for {!r} to {!r}.
                """.format(in_arg, out_arg)))
        elif isinstance(in_arg, np.ndarray):
            if isinstance(out_arg, ug.Ugrid):
                cls = FourierTransformNonuniUni
            elif isinstance(out_arg, np.ndarray):
                cls = FourierTransformNonuniNonuni
            else:
                raise ValueError(errfmt("""\
                No transform defined for {!r} to {!r}.
                """.format(in_arg, out_arg)))

        return super().__new__(cls, *args, **kwargs)

    # Subclasses must override these methods
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError


class FourierTransformUniUni(FourierTransform):
    """Fourier transform on uniformly sampled data. Input is a `Gfunc`
    object. The output is also a `Gfunc` defined on the reciprocal grid.
    Can also be a `Ugrid` or `None`.
    """

    def __init__(self, gfun_in, gfun_out=None, **kwargs):

        # TODO: support different backends
        # TODO: support multithreading
        # TODO: find bottlenecks (profiling)
        # TODO: use self.input_padding parameter

        self.samples = gfun_in
        if gfun_out is not None:
            self.out = gf.asgfunc(gfun_out)  # TODO: Test if this overwrites
        else:
            self.out = None

        # TODO: check these args
        self.input_padding = kwargs.get('input_padding', None)
        self.overwrite_input = kwargs.get('overwrite_input', False)

        old_center = kwargs.get('old_center', None)
        if old_center is not None:
            self.old_center = np.array(old_center)
        else:
            self.old_center = None

        try:
            direction = kwargs.get('direction', 'forward').lower()
        except AttributeError:
            raise ValueError('`direction` must be a string.')

        if direction == 'forward':
            self.direction = direction
            self.execute = self._execute_forward
        elif direction == 'backward':
            self.direction = direction
            self.execute = self._execute_backward
        else:
            raise ValueError(errfmt("""\
            `direction` must be either 'forward' or 'backward'."""))

    def __call__(self):
        return self.execute()

    def _execute_forward(self):
        try:
            import pyfftw as fftw
        except ImportError:
            raise ImportError(errfmt("""\

            Error: The 'pyfftw' module seems to be missing. Please install it
            through your OS's package manager or directly from PyPI:

            https://pypi.python.org/pypi/pyFFTW/
            """))

        if self.out is not None:
            gfun = self.out
        elif self.overwrite_input:
            gfun = self.samples
        else:
            gfun = self.samples.copy()

        self.old_center = gfun.center[:]
        old_spacing = gfun.spacing[:]

        # Pre-process the function values: multiply with
        # (-1)^{sum(index)}
        _mul_isum_parity(gfun)

        # Execute the FFT
        trafo = fftw.builders.fftn(gfun.fvals, overwrite_input=True,
                                   avoid_copy=True,
                                   planner_effort='FFTW_ESTIMATE')
        gfun.fvals = trafo()

        # Transfer to reciprocal grid
        toreciprocal(gfun)

        # Post-process the function values: multiply with
        # (-i)^{sum(shape)} * prod(spacing)
        gfun *= (-1j)**(np.sum(gfun.shape)) * np.prod(old_spacing)
        # (-1)^{sum(index)}
        _mul_isum_parity(gfun)
        # exp(-i<freq,center>)
        even_dims = 1 - np.remainder(gfun.shape, 2)
        if np.any(even_dims) or np.any(self.old_center):
            # TODO: test if the half-shift applies here, too
            # Account for center half-cell shift for even size
            center = self.old_center + old_spacing * even_dims / 2
            _mul_shift_factor(gfun, center)
        # interpol_kerft(freq * spacing)
        _mul_interp_kerft(gfun, old_spacing)

        return gfun

    def _execute_backward(self):
        try:
            import pyfftw as fftw
        except ImportError:
            raise ImportError(errfmt("""\

            Error: The 'pyfftw' module seems to be missing. Please install it
            through your OS's package manager or directly from PyPI:

            https://pypi.python.org/pypi/pyFFTW/
            """))

        if self.out is not None:
            gfun = self.out
        elif self.overwrite_input:
            gfun = self.samples
        else:
            gfun = self.samples.copy()

        # Pre-process the function values: multiply with
        # (-1)^{sum(index)}
        _mul_isum_parity(gfun)
        # exp(i<freq,center>) = exp(-i<freq,-center>)
        even_dims = 1 - np.remainder(gfun.shape, 2)
        if np.any(even_dims) or np.any(self.old_center):
            # TODO: test if the half-shift applies here, too
            # Account for center half-cell shift for even size
            center = (self.old_center[:] if self.old_center is not None
                      else np.zeros(gfun.dim))
            center += even_dims * pi / gfun.tsize
            _mul_shift_factor(gfun, -center)
        # 1 / interpol_kerft(freq * spacing)
        spacing = 2 * pi / gfun.tsize  # location domain spacing
        _div_interp_kerft(gfun, spacing)

        # Execute the FFT
        # NOTE: the inverse transform is normalized such that the factor
        # 1 / prod(shape) is obsolete
        trafo = fftw.builders.ifftn(gfun.fvals, overwrite_input=True,
                                    avoid_copy=True,
                                    planner_effort='FFTW_ESTIMATE')
        gfun.fvals = trafo()

        # Transfer to centeral grid
        gfun.center = self.old_center[:]
        gfun.spacing = 2 * pi / gfun.tsize

        # Post-process the function values: multiply with
        # (-i)^{sum(shape)} / prod(spacing)
        gfun *= (-1j)**(np.sum(gfun.shape)) / np.prod(spacing)
        # (-1)^{sum(index)}
        _mul_isum_parity(gfun)

        return gfun


class FourierTransformUniNonuni(FourierTransform):
    """TODO: doc"""

    def __init__(self, gfun_in, freqs_in, arr_out=None, **kwargs):

        # TODO: support different backends
        # TODO: support multithreading
        # TODO: find bottlenecks (profiling)
        # TODO: use self.input_padding parameter

        if not freqs_in.shape[1] == gfun_in.dim:
            raise InputValidationError(freqs_in.shape[1], gfun_in.dim,
                                       'freqs_in.shape[1]')

        self.samples = gfun_in
        self.freqs = freqs_in

        if arr_out is None:
            self.out = np.empty(self.freqs.shape[0])
        elif arr_out.shape[0] != self.freqs.shape[0]:
            raise InputValidationError(arr_out.shape[0], self.freqs.shape[0],
                                       'arr_out.shape[0]')
        else:
            self.out = arr_out

        # TODO: check?
        self.input_padding = kwargs.get('input_padding', None)

        old_center = kwargs.get('old_center', None)
        if old_center is not None:
            self.old_center = np.array(old_center)
        else:
            self.old_center = None

        try:
            direction = kwargs.get('direction', 'forward').lower()
        except AttributeError:
            raise ValueError('`direction` must be a string.')

        if direction == 'forward':
            self.direction = direction
        elif direction in ('backward', 'adjoint'):
            self.direction = 'adjoint'
        else:
            raise ValueError(errfmt("""\
            `direction` must be either 'forward' or 'backward'/'adjoint'."""))

    def __call__(self):
        return self.execute()

    def execute(self):
        try:
            from pynfft.nfft import NFFT
        except ImportError:
            raise ImportError(errfmt("""\

            Error: The 'pynfft' module seems to be missing. Please install it
            through your OS's package manager or directly from PyPI:

            https://pypi.python.org/pypi/pynfft/
            """))

        if np.any(np.remainder(self.samples.shape, 2)):
            raise ValueError(errfmt("""Number of samples must be even
            in each direction (NFFT restriction)"""))

        # Account for symmetry vs. asymmetry around center (even shape)
        # (gfunc is symmetric, NFFT assumes asymmetric)
        # TODO: check if this is really necessary
        center = self.samples.center + self.samples.spacing / 2

        # Compute shift factor here as `freqs` may be changed later
        if ne:
            freqs = self.freqs
            dotp = ne.evaluate('sum(freqs * center, axis=1)')
            if self.direction == 'forward':
                shift_fac = ne.evaluate('exp(-1j * dotp)')
            else:
                shift_fac = ne.evaluate('exp(1j * dotp)')
        else:
            if self.direction == 'forward':
                shift_fac = np.exp(-1j * np.dot(self.freqs, center))
            else:
                shift_fac = np.exp(1j * np.dot(self.freqs, center))

        # The normalized frequencies must lie between -1/2 and 1/2
        if ne:
            spacing = self.samples.spacing[:]
            norm_freqs = ne.evaluate('''freqs * spacing / (2 * pi)''')
        else:
            norm_freqs = self.freqs * self.samples.spacing / (2 * pi)

        maxfreq = np.max(np.abs(norm_freqs))
        if maxfreq > 0.5:
            raise ValueError('''Frequencies must lie between -0.5 and 0.5
            after normalization, got maximum absolute value: {}.
            '''.format(maxfreq))

        # Initialize the geometry and precompute
        nfft_plan = NFFT(N=self.samples.shape, M=self.freqs.shape[0])
        nfft_plan.x = norm_freqs
        nfft_plan.precompute()

        # Feed in the samples and compute the transform
        if self.direction == 'forward':
            # TODO: check ordering!
            nfft_plan.f_hat = self.samples.fvals.flatten(order='F')
            nfft_plan.trafo()
            self.out = nfft_plan.f

        else:
            # TODO: check ordering!
            nfft_plan.f = self.samples.fvals.flatten(order='F')
            nfft_plan.adjoint()
            self.out = nfft_plan.f_hat

        # Account for shift and scaling
        scaling = (np.prod(self.samples.spacing) /
                   (2 * pi)**(self.samples.dim / 2.))
        if ne:
            out = self.out
            self.out = ne.evaluate('out * shift_fac * scaling')
        else:
            self.out *= shift_fac * scaling

        return self.out


class FourierTransformNonuniUni(FourierTransform):
    pass


class FourierTransformNonuniNonuni(FourierTransform):
    pass


class FourierProjector(object):
    """A base of projectors based on a 'Fourier slice' type theorem.
    See (???) for details.
    It consists in a coordinate transform, two Fourier transforms and two
    graph transforms.
    `Ugrid` for `out` also allowed.
    TODO: more detailed!
    """

    def __new__(cls, *args, **kwargs):
        try:
            in_arg = args[0]
        except IndexError:
            in_arg = kwargs.get('in', None)

        try:
            out_arg = args[1]
        except IndexError:
            out_arg = kwargs.get('out', None)

        if in_arg is None:
            raise ValueError("Unable to determine input type.")

        if isinstance(in_arg, gf.Gfunc):
            if isinstance(out_arg, ug.Ugrid):
                cls = FourierProjectorGfuncGfunc
            elif isinstance(out_arg, np.ndarray):
                cls = FourierProjectorGfuncGraph
            else:
                raise ValueError(errfmt("""\
                No projector defined for {!r} to {!r}.
                """.format(in_arg, out_arg)))
        elif isinstance(in_arg, np.ndarray):
            if isinstance(out_arg, gf.Gfunc):
                cls = FourierProjectorGraphGfunc
            elif isinstance(out_arg, np.ndarray):
                cls = FourierProjectorGraphGraph
            else:
                raise ValueError(errfmt("""\
                No projector defined for {!r} to {!r}.
                """.format(in_arg, out_arg)))

        return super().__new__(cls, *args, **kwargs)

    # Subclasses must override these methods
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError


class FourierProjectorGfuncGfunc(FourierProjector):

        def __init__(self, samples, projection, coord_trafo, **kwargs):

            # TODO: how to check consistency?
            self.coord_trafo = coord_trafo
            self.freq_gr_trafo = kwargs.get('freq_graph_trafo', None)
            self.final_gr_trafo = kwargs.get('final_graph_trafo', None)

            self.overwrite_proj = kwargs.get('overwrite_proj', False)

            self.samples = samples

            if isinstance(projection, gf.Gfunc):
                self.out = (projection if self.overwrite_proj
                            else projection.copy())
            else:
                if self.overwrite_proj:
                    raise ValueError(errfmt("""\
                    'overwrite_proj' not possible for
                    type(projection) == ugrid.Ugrid"""))
                self.out = gf.asgfunc(projection)

        def __call__(self):
            return self.execute()

        def execute(self):
            proj = self.out

            # Transform the frequency coordinates
            old_orig = proj.center[:]
            proj.toreciprocal()
            freqs = self.coord_trafo(proj.coord)

            # Debugging
            from utility import plot3d_scatter
            plot3d_scatter(freqs)

            # Compute the Fourier Trafo of the samples at the frequencies
            ft = FourierTransform(self.samples, freqs)
            # TODO: re-order or warn the user!
            ftvals = ft()

            # Apply the frequency space graph trafo
            if self.freq_gr_trafo is not None:
                print('Apply graph trafo!')
                ftvals = self.freq_gr_trafo(freqs, ftvals)

            # Write FT values back to `proj` and compute inverse FT
            # TODO: check ordering!
            proj.fvals = ftvals.reshape(proj.shape, order='F')
            print('before ift:\n')
            proj.display()

            ift = FourierTransform(proj, direction='backward',
                                   old_center=old_orig)

            proj = ift()
            print('after ift:\n')
            proj.display()

            # Apply the final (real space) graph trafo
            if self.final_gr_trafo is not None:
                proj.fvals = self.final_gr_trafo(proj.asgraph())

            return proj


class FourierProjectorGraphGfunc(FourierProjector):
        pass


class FourierProjectorGfuncGraph(FourierProjector):
        pass


class FourierProjectorGraphGraph(FourierProjector):
        pass


def reciprocal(grid):
    """Return the reciprocal grid of the input. By convention, it always
    contains zero, i.e. the center is shifted by half a grid cell in
    even dimensions.
    """
    even_dims = 1 - np.remainder(grid.shape, 2)
    return ug.Ugrid(grid.shape, -np.pi / grid.tsize * even_dims,
                    2 * np.pi / grid.tsize)


def toreciprocal(grid):
    """Turn the input grid into its reciprocal. By convention, it always
    contains zero, i.e. the center is shifted by half a grid cell in
    even dimensions.
    """
    even_dims = 1 - np.remainder(grid.shape, 2)
    grid.center = -np.pi / grid.tsize * even_dims
    grid.spacing = 2 * np.pi / grid.tsize


def _mul_isum_parity(gfun):
    negate = lambda val: -val
    oddind_slc = np.s_[1::2]
    for i in range(gfun.dim):
        gfun.apply_0dmapping_with_1dslice(negate, axis=i, ax_slc=oddind_slc)


def _mul_shift_factor(gfun, center):
    from cmath import exp
    expmap = lambda xi, val, org: val * exp(-1j * xi * org)
    for i in range(gfun.dim):
        expmapi = partial(expmap, org=center[i])
        gfun.apply_1dmapping(expmapi, axis=i)


def _mul_interp_kerft(gfun, spacing):
    from math import cos

    def kerft_map(freq, val, spc):
        arg = freq * spc
        if abs(arg) > 0.05:
            return val * (2 * (1 - cos(arg)) / arg**2) / SQRT_2PI
        else:
            return val * ((1. - arg**2 / 12. + arg**4 / 360. - arg**6 / 20160.)
                          / SQRT_2PI)

    for i in range(gfun.dim):
        kerft_mapi = partial(kerft_map, spc=spacing[i])
        gfun.apply_1dmapping(kerft_mapi, axis=i)


def _div_interp_kerft(gfun, spacing):
    from math import cos

    def kerft_map(freq, val, spc):
        arg = freq * spc
        if abs(arg) > 0.05:
            return val * arg**2 / (2 * (1 - cos(arg))) * SQRT_2PI
        else:
            return val * ((1. + arg**2 / 12. + arg**4 / 240. + arg**6 / 6048.)
                          * SQRT_2PI)

    for i in range(gfun.dim):
        kerft_mapi = partial(kerft_map, spc=spacing[i])
        gfun.apply_1dmapping(kerft_mapi, axis=i)
