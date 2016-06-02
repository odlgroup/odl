from __future__ import print_function, division, absolute_import
from future import standard_library


import sys
standard_library.install_aliases()

# External
import numpy as np
import pytest
from odl.tomo.backends.stir_setup import STIR_AVAILABLE

import odl
from odl.util.testutils import is_subdict

pytestmark = pytest.mark.skipif("not odl.tomo.STIR_AVAILABLE")


def test_vol_geom_3d():
    """ Create STIR 3D volume."""

    # Create a domain compatible to STIR
    discr_dom_odl = odl.tomo.stir_get_ODL_domain_which_honours_STIR_restrictions([151, 151, 100], [2.5, 2.5, 1])

    # Use the aforementioned domain as a template for STIR domain
    stir_discr_dom = odl.tomo.stir_get_STIR_domain_from_ODL(discr_dom_odl)

    # Use the STIR domain as a template to create a new ODL domain
    new_discr_dom_odl = odl.tomo.stir_get_ODL_domain_from_STIR(stir_discr_dom)

    # I don't know if this util suppose to work this way.
    return odl.util.testutils.all_almost_equal(discr_dom_odl, new_discr_dom_odl)


def test_vector_transfer(_display = False):
    """
    Test if the interface handles vectors correctly.
    Several things are going to be tested here:
    . Add ot multiply of a vector from ODL to STIR
    . Compare the two copies
    . Print functions
    Returns
    -------
    True or False

    """
    # Create a domain compatible to STIR
    discr_dom_odl = odl.tomo.stir_get_ODL_domain_which_honours_STIR_restrictions([151, 151, 151], [2.5, 2.5, 2.5])

    # A sample phantom in odl domain
    odl_phantom = odl.util.shepp_logan(discr_dom_odl, modified=True)

    # Create an empty STIR phantom based on the ODL domain
    stir_phantom = odl.tomo.stir_get_STIR_domain_from_ODL(discr_dom_odl)

    odl_phantom_array = odl_phantom.asarray()

    # Display starting array
    if _display:
        fig1 = odl.tomo.stir_unified_display_function(odl_phantom_array[:,:,70], odl_phantom, _title="Initial ODL phantom")

    # Operate ODL data on STIR data
    odl.tomo.stir_operate_STIR_and_ODL_vectors(stir_phantom, odl_phantom, '+')

    stir_phantom_array = odl.tomo.stir_get_STIR_data_as_array(stir_phantom)

    if _display: # Display intermediate array
        fig3 = odl.tomo.stir_unified_display_function(stir_phantom_array[70,:,:],odl_phantom, _title="STIR phantom")

    return True

if __name__ == '__main__':
    # pytest.main(str(__file__.replace('\\', '/')) + ' -v')
    print("1. Create a ODL domain (with restrictions) : Trasform it to STIR : Reverse transform back to ODL and match. -> ", test_vol_geom_3d())
    print("\n")
    print("2. Create a ODL vector : Trasform it to STIR : Reverse transform back to ODL and match. -> ", test_vector_transfer(_display=True))
