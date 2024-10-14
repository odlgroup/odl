###############################################################################
#          This code was taken from tomosipo and adapted to ODL API           #
#            Please check https://github.com/ahendriksen/tomosipo             #
###############################################################################
"""ASTRA conversion and projection code

There are two geometry conversion methods:

- from_astra
- to_astra

An important method is `create_astra_projector`, which creates an ASTRA
projector from a pair of geometries.

Moreover, there is projection code that is centered around the following
ASTRA APIs:

1. astra.experimental.direct_FPBP3D (modern)
2. astra.experimental.do_composite (legacy)

The first is used in modern tomosipo code: it takes an existing ASTRA projector
and a link to a numpy or gpu array.

The second is a legacy approach that is kept for debugging and testing purposes.
It takes multiple Data objects describing volumes (both data and geometry) and
projection geometries (both data and geometry). On this basis, it creates a
projector and passes it to ASTRA, which performs an all-to-all (back)projection.

"""
import astra.experimental as experimental
from odl.tomo.backends import links

###############################################################################
#                       Direct ASTRA projection (modern)                      #
###############################################################################
def direct_project(
    projector,
    vol_link,
    proj_link,
    forward=None,
    additive=False,
):
    """Project forward or backward

    Forward/back projects a volume onto a projection dataset.

    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector` to generate an astra projector.
    :param vol_link: TODO
    :param proj_link: TODO
    :param forward: bool
        True for forward project, False for backproject.
    :param additive: bool
        If True, add projection data to existing data. Otherwise
        overwrite data.
    :returns:
    :rtype:

    """
    if forward is None:
        raise ValueError("project must be given a forward argument (True/False).")

    # These constants have become the default. See:
    # https://github.com/astra-toolbox/astra-toolbox/commit/4d673b3cdb6d27d430087758a8081e4a10267595
    MODE_SET = 1
    MODE_ADD = 0

    if not links.are_compatible(vol_link, proj_link):
        raise ValueError(
            "Cannot perform ASTRA projection on volume and projection data, because they are not compatible. "
            "Usually, this indicates that the data are located on different computing devices. "
        )

    # If necessary, the link may adjust the current state of the
    # program temporarily to ensure ASTRA runs correctly. For torch
    # tensors, this may entail changing the currently active GPU    
    with vol_link.context():
        experimental.direct_FPBP3D( #type:ignore
            projector,
            vol_link.linked_data,
            proj_link.linked_data,
            MODE_ADD if additive else MODE_SET,
            "FP" if forward else "BP",
        )

def direct_fp(
    projector,
    vol_data,
    proj_data,
    additive=False,
):
    """Project forward or backward

    Forward/back projects a volume onto a projection dataset.

    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector` to generate an astra projector.
    :param vol_data: TODO
    :param proj_data: TODO
    :param additive: bool
        If True, add projection data to existing data. Otherwise
        overwrite data.
    :returns:
    :rtype:

    """
    return direct_project(
        projector,
        vol_data,
        proj_data,
        forward=True,
        additive=additive,
    )


def direct_bp(
    projector,
    vol_data,
    proj_data,
    additive=False,
):
    """Project forward or backward

    Forward/back projects a volume onto a projection dataset.

    :param projector: ??
        It is possible to provide a pre-generated ASTRA projector. Use
        `ts.Operator.astra_projector` to generate an astra projector.
    :param vol_data: TODO
    :param proj_data: TODO
    :param additive: bool
        If True, add projection data to existing data. Otherwise
        overwrite data.
    :returns:
    :rtype:

    """
    return direct_project(
        projector,
        vol_data,
        proj_data,
        forward=False,
        additive=additive,
    )

