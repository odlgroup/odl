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

"""Test for X-ray transforms."""

# # Imports for common Python 2/3 codebase
# from __future__ import print_function, division, absolute_import
# from future import standard_library
#
# standard_library.install_aliases()
#
# # External
# import numpy as np
#
# # Internal
# from odl.discr.lp_discr import uniform_discr, uniform_sampling
# from odl.set.domain import Interval, Rectangle
# from odl import CircularConeFlatGeometry, DiscreteXrayTransform

# discr_reco_space = uniform_discr(
#         [-20, -20, -20], [20, 20, 20], [300, 300, 300], dtype='float32')
#
# src_rad = 1000
# det_rad = 100
# angle_intvl = Interval(0, 2 * np.pi)
# dparams = Rectangle([-50, -50], [50, 50])
# agrid = uniform_sampling(angle_intvl, 360, as_midp=False)
# dgrid = uniform_sampling(dparams, [558, 558])
# geom = CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad, agrid,
#                                 dgrid)
#
# xray_trafo = DiscreteXrayTransform(discr_reco_space, geom,
#                                    backend='astra_cuda')
#
# discr_vol_data = xray_trafo.domain.one()
#
# # X-ray transform
# proj_data = xray_trafo(discr_vol_data)
#
# proj_data = xray_trafo.range.element(np.zeros((360, 558, 558)))
#
# # back projection
# back_proj = xray_trafo.adjoint(proj_data)
