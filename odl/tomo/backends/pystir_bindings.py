# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.util.graphics import show_discrete_data
from odl.discr.grid import RegularGrid

import matplotlib.pyplot as plt

try:
    import pylab
    HAVE_PYLAB = True
except ImportError:
    HAVE_PYLAB = False

import sys
import numpy as np

# These are the local paths were pSTIR and cSTIR exist.
sys.path.append("/home/lcrguest/dev/CCPPETMR/xSTIR/pSTIR")
sys.path.append("/home/lcrguest/dev/CCPPETMR/xSTIR/cSTIR")

try:
    # Since there is a global stir.py I renamed the local version to _pstir
    import _pstir as stir
    PSTIR_AVAILABLE = True
except ImportError:
    PSTIR_AVAILABLE = False

__all__ = ('PSTIR_AVAILABLE', 'pstir_get_volume_geometry',
           'pstir_copy_DiscreteLP','pstir_copy_DiscreteLPVector',
           'check_consistency_between_domain_and_geometry')


from odl.discr.lp_discr import DiscreteLp, DiscreteLpVector
from odl.space.ntuples import FnVector
from odl.tomo.geometry import (\
    Geometry, Parallel2dGeometry, DivergentBeamGeometry, ParallelGeometry,\
    FlatDetector)

def pstir_get_volume_geometry(discr_reco):

    vol_shp = discr_reco.partition.shape
    voxel_size = discr_reco.cell_sides

    if discr_reco.ndim == 2:
        return vol_shp + (1,), tuple(voxel_size) + (0.1,)
    else:
        return vol_shp, tuple(voxel_size)

def pstir_copy_DiscreteLP(discr_reco):
    """
    Copies the geometry of a DiscretisedLP object.
    In STIR this would make an empty densitity or sensitivity image.
    Parameters
    ----------
    discr_reco: A 2D or 3D DiscretisedLP object

    Returns: A STIR volume
    -------

    """

    if not isinstance(discr_reco, DiscreteLp):
        raise TypeError('discretized domain {!r} is not a DiscreteLp '
                        'instance.'.format(discr_reco))

    im_dim, vox_size = pstir_get_volume_geometry(discr_reco)
    image = stir.Image()
    image.initialise(im_dim, vox_size)

    return image


def pstir_copy_DiscreteLPVector(discr_reco, _data):
    """
    It uses a DiscretisedLP object to obtain the volume characteristics and then
    copies the data from a DiscretisedLPVector. The result is a STIR volume with some
    data.
    Parameters
    ----------
    discr_reco: A 2D or 3D DiscretisedLP object
    data: A 2D or 3D DiscretisedLPVector object

    Returns
    -------
    A STIR volume with some data.
    """

    if not isinstance(discr_reco, DiscreteLp):
        raise TypeError('discretized domain {!r} is not a DiscreteLp '
                        'instance.'.format(discr_reco))

    im_dim, vox_size = pstir_get_volume_geometry(discr_reco)
    image = stir.Image()
    image.initialise(im_dim, vox_size)

    vol_min = discr_reco.partition.begin
    vol_max = discr_reco.partition.end

    fig1 = plt.figure()
    _data.show(indices=np.s_[:,:,150], fig=fig1)
    fig1.canvas.set_window_title('ODL phantom')

    dn = _data.asarray()

    # Transform the array to what STIR needs
    dn2 = np.rot90(dn,-1)
    dn3 = np.fliplr(dn2)
    # STIR indices are [z, y, x]
    dn4 = np.swapaxes(dn3,0,2)
    #

    dn5 = dn4.copy()

    fig2 = plt.figure()
    grid = RegularGrid([vol_min[0], vol_min[1]], [vol_max[0], vol_max[1]], (2, 3))
    show_discrete_data(dn4[150,:,:], grid, fig = fig2)
    fig2.canvas.set_window_title('STIR input')

    # pylab.imshow(dn2[150,:,:])
    # pylab.show()

    image.cSTIR_addVector(dn5)

    data = image.as_array()
    fig3 = plt.figure()
    grid = RegularGrid([vol_min[0], vol_min[1]], [vol_max[0], vol_max[1]], (2, 3))
    show_discrete_data(data[150,:,:], grid, fig = fig3)
    fig3.canvas.set_window_title('Actual STIR image')

    nikos = 0

    return image

def check_consistency_between_domain_and_geometry():
    pass