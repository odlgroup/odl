from __future__ import print_function, division, absolute_import
from future import standard_library

import odl

standard_library.install_aliases()

import sys
import numpy as np
import matplotlib.pyplot as plt

from odl.util.graphics import show_discrete_data
from odl.discr.grid import RegularGrid

from odl.discr.lp_discr import DiscreteLp, DiscreteLpVector
from odl.space.ntuples import FnVector
from odl.tomo.geometry import (\
    Geometry, Parallel2dGeometry, DivergentBeamGeometry, ParallelGeometry,\
    FlatDetector, Flat2dDetector)
from odl.tomo.util.utility import perpendicular_vector


# These are the local paths were pSTIR and cSTIR exist.
sys.path.append("/home/lcrguest/dev/CCPPETMR/xSTIR/pSTIR")
sys.path.append("/home/lcrguest/dev/CCPPETMR/xSTIR/cSTIR")
sys.path.append("/home/lcrguest/dev/CCPPETMR/STIR_build/src/swig")
sys.path.append("/home/lcrguest/dev/CCPPETMR/STIR/src/swig")

try:
    import stir
    import _pstir
    # Fix for stirextra being moved around in various stir versions
    try:
        stirextra = stir.stirextra
    except AttributeError:
        import stirextra

    STIR_AVAILABLE = True
except ImportError:
    STIR_AVAILABLE = False


__all__ = ('STIR_AVAILABLE',
        'pstir_transform_DiscreteLPVector_to_STIR_compatible_Image',\
        'pstir_get_ODL_geoemtry_which_honours_STIR_restrictions',\
        'pstir_get_STIR_domain_from_ODL', 'pstir_get_ODL_domain_from_STIR',\
        'pstir_get_STIR_empty_array_from_ODL',
        'pstir_get_ODL_domain_which_honours_STIR_restrictions',
        'pstir_transform_array_to_STIR_compatible_array',
        'pstir_unified_display_function' )


#
#
# INTERFACE FUNCTIONS FROM ODL TO STIR
#

def pstir_get_ODL_domain_which_honours_STIR_restrictions(_vox_num, _vox_size):
    """
    In the future a geometry should be imported to handle scanner alignment restrictions.
    Returns
    -------

    .. warning:: STIR coordinates are currently related to the scanner, i.e. not to the patient (as in DICOM).
        For an image with zero offset, the origin is assumed to coincide with the centre of the first plane.
        I strongly suggest for the time being to let the origin default to (0,0,0) as STIR regards it.


    """

    range = [a*b for a,b in zip(_vox_num,_vox_size)]
    min_p = [-x / 2 for x in range]
    max_p = [a+b for a,b in zip(min_p,range)]

    min_p[2] = 0.0
    max_p[2] = range[2]

    return odl.uniform_discr(
            min_corner=min_p, max_corner=max_p, nsamples= _vox_num,
            dtype='float32')


def pstir_get_ODL_geoemtry_which_honours_STIR_restrictions(_det_y_size_mm, _det_z_size_mm,\
                                                           _num_rings, _num_dets_per_ring,\
                                                           _det_radius):
    if _det_radius <= 0:
        raise ValueError('ring circle radius {} is not positive.'
                             ''.format(_det_radius))

    axis=[0, 0, 1]

    first_tmpl_det_point = [-_det_y_size_mm/2, -_det_z_size_mm/2]
    last_tmpl_det_point = [_det_y_size_mm/2, _det_z_size_mm/2]

    # Template of detectors
    tmpl_det = odl.uniform_partition(first_tmpl_det_point,\
                                         last_tmpl_det_point,\
                                         [1, 1])

    # Perpendicular vector from the ring center to
    # the first detector of the same ring
    ring_center_to_tmpl_det = perpendicular_vector(axis)

    ring_center_to_det = (np.array(ring_center_to_tmpl_det) /
                                     np.linalg.norm(ring_center_to_tmpl_det))

    det_init_axis_0 = np.cross(axis, ring_center_to_det)
    det_init_axes = (det_init_axis_0, axis)

    apart = odl.uniform_partition(0, 2 * np.pi, _num_dets_per_ring,
                                      nodes_on_bdry=True)

    detector = Flat2dDetector(tmpl_det, det_init_axes)

    # Axial (z-axis) movement parameters.
    # The middle of the first ring should be on (r,r,0.0)
    axialpart = odl.uniform_partition(0, _num_rings*_det_z_size_mm, _num_rings)


    return odl.tomo.geometry.CylindricalPetGeom(_det_radius,
                                                tmpl_det,
                                                ring_center_to_det,
                                                apart,
                                                detector,
                                                axialpart)



def pstir_get_STIR_domain_from_ODL(_discreteLP):
    """
    Interface function to get a STIR domain without caring about the classes names.

    Parameters
    ----------
    _discreteLP

    Returns
    -------

    """

    return pstir_create_VoxelsOnCartesianGrid_from_DiscreteLP(_discreteLP)


def pstir_get_ODL_domain_from_STIR(_voxels):
    """
    Interface function to get an ODL domain without caring about the classes names.

    Parameters
    ----------
    _voxelsF

    Returns
    -------

    """
    return pstir_create_DiscreteLP_from_STIR_voxels(_voxels)


def pstir_get_STIR_empty_array_from_ODL(_discreteLP):
    """
    An array of data in STIR represented by Array<int,size_t>. It can store data but
    not physical sizes. In pySTIR the array class is interfaced by the Image class (which can
      actually be a volume).

    Parameters
    ----------
    _discreteLP

    Returns
    -------

    """
    return pstir_create_empty_Image_array_from_DiscreteLP(_discreteLP)

def pstir_transform_DiscreteLPVector_to_STIR_compatible_Image(_domain, _data):
    """
    This function groups the complete process.
    Parameters
    ----------
    _domain
    _data

    Returns
    -------

    """
    data_array = _data.asarray()

    trans_data_array = pstir_transform_array_to_STIR_compatible_array(data_array)

    stir_phantom = pstir_get_STIR_empty_array_from_ODL(_domain)
    stir_phantom.cSTIR_addVector(trans_data_array)

    return stir_phantom


def pstir_get_STIR_scanner_from_ODL_geometry(_geometry):

    scanner = stir.Scanner()

    stir.Scanner.set_average_depth_of_interaction(_geometry.stir_average_depth_of_inter)
    stir.Scanner.set_default_bin_size(_geometry.stir_bin_size)
    stir.Scanner.set_default_intrinsic_tilt(_geometry.stir_intrinsic_tilt)
    stir.Scanner.set_default_bin_size(_geometry.stir_max_num_non_arccorrected_bins_v)
    stir.Scanner.set_inner_ring_radius(_geometry.stir_inner_ring_rad)
    stir.Scanner.set_max_num_non_arccorrected_bins(_geometry.stir_max_num_non_arccorrected_bins)
    stir.Scanner.set_num_axial_blocks_per_bucket(_geometry.stir_num_axial_crystals_per_block)
    stir.Scanner.set_num_axial_crystals_per_block(_geometry.stir_num_axial_crystals_per_singles_unit)
    stir.Scanner.set_num_detector_layers(_geometry.stir_num_detector_layers)

#
#
# FUNCTIONS FROM STIR TO ODL
#

#
#
# TRANSFORM FUNCTIONS
#

def pstir_transform_array_to_STIR_compatible_array(_this_array):
    """
    This is transformation function. The input is a numpy array from a DiscreteLPVector and returns
    an array which can be used to fill a STIR image and maintain structure.
    Parameters
    ----------
    _this_array: A NumPy array from a DiscreteLPVector

    Returns
    -------
    A NumPy array compatible to STIR

    """

    _this_array = np.rot90(_this_array,-1)
    _this_array = np.fliplr(_this_array)
    # STIR indices are [z, y, x]
    _this_array = np.swapaxes(_this_array,0,2)

    # I have to copy in order to transform the actual data
    return _this_array.copy()


#
#
# INFO / DISPLAY FUNCTIONS
#


def pstir_display_STIR_image(_this_image, _this_grid):
    pass



def pstir_display_array():
    pass


#
#
# HELPERS
#

def pstir_get_volume_geometry(discr_reco):
    """
    This is a helper function which returns the total size and voxel number of a
    discretised object.

    Parameters
    ----------
    discr_reco: A discretised ODL object

    Returns
    -------



    """
    vol_shp = discr_reco.partition.shape
    voxel_size = discr_reco.cell_sides

    return vol_shp, tuple(voxel_size)


def pstir_create_VoxelsOnCartesianGrid_from_DiscreteLP(_discr):
    """
    This class is used to represent voxelised densities on a cuboid grid (3D).
    This class represents 'normal' data. Basisfunctions are just voxels.
    ----------
    _discr: DiscreteLP

    Returns
    -------
    An empty STIR image object

    """

    if not isinstance(_discr, DiscreteLp):
        raise TypeError('discretized domain {!r} is not a DiscreteLp '
                        'instance.'.format(discr_reco))

    if _discr.ndim == 2:
        raise Exception('The STIR (domain) image should be a 3D object')

    if _discr.partition.begin[2] < 0.0:
        raise Exception('STIR z axis first index must be on 0.0 mm, got {!r} mm'
                        .format(_discr.partition.begin[2]))

    im_dim, vox_size = pstir_get_volume_geometry(_discr)

    return _pstir.Voxels(im_dim, vox_size)


def pstir_create_empty_Image_array_from_DiscreteLP(_discr):
    """
    This class defines multi-dimensional (numeric) arrays.
    This class implements multi-dimensional arrays which can have 'irregular' ranges.
    See IndexRange for a description of the ranges. Normal numeric operations are defined.
    In addition, two types of iterators are defined, one which iterators through the outer index,
    and one which iterates through all elements of the array.

    Array inherits its numeric operators from NumericVectorWithOffset.
    In particular this means that operator+= etc. potentially grow the object.
    However, as grow() is a virtual function,
    Array::grow is called, which initialises new elements first to 0.

    Parameters
    ----------
    _discr

    Returns
    -------

    .. warning: Currently the python support of this class is quite limited and
     underlying assumptions could be made.

    """

    im_dim, vox_size = pstir_get_volume_geometry(_discr)

    image = _pstir.Image()
    image.initialise(im_dim, vox_size)

    return image


def pstir_create_DiscreteLP_from_STIR_voxels(_voxels):
    """
    This function tries to transform the VoxelsOnCartesianGrid to
    DicreteLP.

    Parameters
    ----------
    _voxels: A VoxelsOnCartesianGrid Object

    Returns
    -------

    .. warning:: Still there problems with the node location in each pixel.
    .. todo:: I must work in the domain size misalignment.
    """
    idims, fdims = _voxels.get_physical_dimensions()

    vol_shp = [idims[2], idims[1], idims[0]]

    # Trying the avoid the mesh from the different precisions.
    strings = np.array(["{:10.6f}".format(number)  for number in fdims])
    fdims2 = strings.astype(np.float64)

    # STIR arrays : [z][y][x]
    vol_min = [fdims2[2], fdims2[1], fdims2[0]]
    vol_max = [fdims2[5], fdims2[4], fdims2[3]]

    return odl.uniform_discr(
            min_corner=vol_min, max_corner=vol_max, nsamples=vol_shp,
            dtype='float32')


def pstir_unified_display_function(_display_me, _in_this_grid, _title=""):
    """
    This is a helper function. STIR, ODL and NumPy used different functions to display images.
    I created this function, in order to avoid flips and rotates.
    It which calls odl.utils.graphics.show_discrete_data.

    Parameters
    ----------
    _display_me: A NumPy array.

    _in_this_grid: A suitable grid

    _title: A title for the figure

    Returns
    -------
    A matplotlib.pyplot figure
    """

    grid = get_2D_grid_from_domain(_in_this_grid)

    fig = plt.figure()
    show_discrete_data(_display_me, grid, fig = fig)
    fig.canvas.set_window_title(_title)


def get_2D_grid_from_domain(_this_domain):
    """

    Parameters
    ----------
    _this_domain

    Returns
    -------

    """
    return  RegularGrid([_this_domain.space.partition.begin[0], _this_domain.space.partition.begin[1]],
                        [_this_domain.space.partition.end[0], _this_domain.space.partition.end[1]], (2, 3))