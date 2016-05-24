from __future__ import print_function, division, absolute_import
from future import standard_library


import sys
standard_library.install_aliases()
# These are the local paths were pSTIR and cSTIR exist.
sys.path.append("/home/lcrguest/dev/CCPPETMR/xSTIR/pSTIR")
sys.path.append("/home/lcrguest/dev/CCPPETMR/xSTIR/cSTIR")

# External
import numpy as np
import pytest
from odl.tomo.backends.stir_setup import STIR_AVAILABLE

if STIR_AVAILABLE:
    import _pstir as stir

import odl
from odl.util.testutils import is_subdict

pytestmark = pytest.mark.skipif("not odl.tomo.STIR_AVAILABLE")


def test_vol_geom_3d():
    """ Create STIR 3D volume."""

    # Create a domain compatible to STIR
    discr_dom_odl = odl.tomo.pstir_get_ODL_domain_which_honours_STIR_restrictions([151, 151, 151], [2.5, 2.5, 2.5])

    # Use the aforementioned domain as a template for STIR domain
    stir_discr_dom = odl.tomo.pstir_get_STIR_domain_from_ODL(discr_dom_odl)

    # Use the STIR domain as a template to create a new ODL domain
    new_discr_dom_odl = odl.tomo.pstir_get_ODL_domain_from_STIR(stir_discr_dom)

    # I don't know if this util suppose to work this way.
    return odl.util.testutils.all_almost_equal(discr_dom_odl, new_discr_dom_odl)


def test_vector_transfer(_display = False):
    """
    Test if the interface handles vectors correctly.
    Several things are going to be tested here:
    . Transfer of a vector from ODL to STIR
    . Write it to HDD
    . Load it from HDD
    . Compare the two copies
    . Reverse transform it to ODL
    . Compare the two copies
    . Print functions
    Returns
    -------
    True or False

    """
    # Create a domain compatible to STIR
    discr_dom_odl = odl.tomo.pstir_get_ODL_domain_which_honours_STIR_restrictions([151, 151, 151], [2.5, 2.5, 2.5])

    # A sample phantom in odl domain
    phantom = odl.util.shepp_logan(discr_dom_odl, modified=True)
    # phantom.show(indices=np.s_[:,:,70])

    # Create an empty STIR phantom based on the ODL domain
    stir_phantom = odl.tomo.pstir_get_STIR_empty_array_from_ODL(discr_dom_odl)

    # Before transfer data to STIR they have to be transformed in np.array
    # - This can be handled by the underlying functions but I need it now for display purposes.
    phantom_array = phantom.asarray()

    if _display: # Display starting array
        fig1 = odl.tomo.pstir_unified_display_function(phantom_array[:,:,70], phantom, _title="Initial ODL phantom")

    # Transform the array to STIR axes and orientation
    trans_phantom_array = odl.tomo.pstir_transform_array_to_STIR_compatible_array(phantom_array)

    if _display: # Display intermediate array
        fig2 = odl.tomo.pstir_unified_display_function(trans_phantom_array[:,:,70],phantom, _title="Transformed ODL phantom")

    # Fill the STIR image with data from the phantom_array - NOTE: This class has different coordinates, see the display slice selection
    stir_phantom.cSTIR_addVector(trans_phantom_array)
    ff = stir_phantom.as_array()

    if _display: # Display intermediate array
        fig3 = odl.tomo.pstir_unified_display_function(ff[70,:,:],phantom, _title="STIR phantom")






     pause = True


if __name__ == '__main__':
    # pytest.main(str(__file__.replace('\\', '/')) + ' -v')
    # print("1. Create a ODL domain (with restrictions) : Trasform it to STIR : Reverse transform back to ODL and match. -> ", test_vol_geom_3d())
    print("\n")
    test_vector_transfer(_display=True)