"""Example PET reconstruction using STIR.

This example computes projections from the ODL Shepp-Logan phantom
and uses them as input data for reconstruction in STIR. Definition
of the acquisition geometry and computations are done entirely in STIR,
where the communication between ODL and STIR is realized with files
via hard disk.

Note that running this example requires an installation of
`STIR <http://stir.sourceforge.net/>`_ and its Python bindings.
"""

from os import path

import odl

# Set path to input files
base = path.join(path.dirname(path.abspath(__file__)), 'data', 'stir')
volume_file = str(path.join(base, 'initial.hv'))
projection_file = str(path.join(base, 'small.hs'))

# Create a STIR projector from file data.
proj = odl.tomo.backends.stir_bindings.stir_projector_from_file(
    volume_file, projection_file)

# Create Shepp-Logan phantom
vol = odl.phantom.shepp_logan(proj.domain, modified=True)

# Project data. Note that this delegates computations to STIR.
projections = proj(vol)

# Calculate operator norm required for Landweber's method
op_norm_est_squared = proj.adjoint(projections).norm() / vol.norm()
omega = 0.5 / op_norm_est_squared

# Reconstruct using the STIR forward projector in the ODL reconstruction scheme
recon = proj.domain.zero()
odl.solvers.landweber(proj, recon, projections, niter=50, omega=omega)
recon.show(force_show=True)
