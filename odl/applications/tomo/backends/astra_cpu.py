# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Backend for ASTRA using CPU."""


import warnings

import numpy as np

from odl.core.discr import DiscretizedSpace, DiscretizedSpaceElement
from odl.applications.tomo.backends.astra_setup import (
    astra_algorithm, astra_data, astra_projection_geometry, astra_projector,
    astra_volume_geometry)
from odl.applications.tomo.backends.util import _add_default_complex_impl
from odl.applications.tomo.geometry import (
    DivergentBeamGeometry, Geometry, ParallelBeamGeometry)
from odl.core.util import writable_array
from odl.core.array_API_support import lookup_array_backend, get_array_and_backend

try:
    import astra
except ImportError:
    pass

__all__ = (
    'astra_cpu_projector',
    'default_astra_proj_type',
)


def default_astra_proj_type(geom):
    """Return the default ASTRA projector type for a given geometry.

    Parameters
    ----------
    geom : `Geometry`
        ODL geometry object for which to get the default projector type.

    Returns
    -------
    astra_proj_type : str
        Default projector type for the given geometry.

        In 2D:

        - `ParallelBeamGeometry`: ``'linear'``
        - `DivergentBeamGeometry`: ``'line_fanflat'``

        In 3D:

        - `ParallelBeamGeometry`: ``'linear3d'``
        - `DivergentBeamGeometry`: ``'linearcone'``
    """
    if isinstance(geom, ParallelBeamGeometry):
        return 'linear' if geom.ndim == 2 else 'linear3d'
    elif isinstance(geom, DivergentBeamGeometry):
        return 'line_fanflat' if geom.ndim == 2 else 'linearcone'
    else:
        raise TypeError(
            f"no default exists for {type(geom)}, `astra_proj_type` must be given explicitly"
        )


def astra_cpu_projector(
        direction:str,
        input_data:DiscretizedSpaceElement,
        geometry:Geometry,
        range_space:DiscretizedSpace,
        out :DiscretizedSpaceElement = None,
        astra_proj_type: str | None = None
        ) -> DiscretizedSpaceElement:
    """Run an ASTRA projection on the given data using the CPU.

    Parameters
    ----------
    input_data : `DiscretizedSpaceElement`
        Input data to which the projector is applied.
    geometry : `Geometry`
        Geometry defining the tomographic setup.
    range_space : `DiscretizedSpace`
        Space to which the calling operator maps.
    out : ``range_space`` element, optional
        Element of the range_space space to which the result is written. If
        ``None``, an element in ``range`` is created.
    astra_proj_type : str, range_space
        Type of projector that should be used. See `the ASTRA documentation
        <http://www.astra-toolbox.com/docs/proj2d.html>`_ for details.
        By default, a suitable projector type for the given geometry is
        selected, see `default_astra_proj_type`.

    Returns
    -------
    out : ``proj_space`` element
        Projection data resulting from the application of the projector.
        If ``out`` was provided, the returned object is a reference to it.
    """
    ### Asserting that we get the right data types.
    assert direction in ['forward', 'backward']
    if not isinstance(input_data, DiscretizedSpaceElement):
        raise TypeError(
            f"Input data {input_data} is not a `DiscretizedSpaceElement` instance"
        )
    if not isinstance(geometry, Geometry):
        raise TypeError(f"geometry {geometry} is not a Geometry instance")
    if not isinstance(range_space, DiscretizedSpace):
        raise TypeError(
            f"`range_space` {range_space} is not a DiscretizedSpace instance"
        )
    if input_data.ndim != geometry.ndim:
        raise ValueError(
            f"dimensions {input_data} of input data and {geometry.ndim} of geometry do not match"
        )
    if out is None:
        out_element = range_space.real_space.element()
    else:
        if out not in range_space.real_space:
            raise TypeError(
                f"`out` {out} is neither None nor a `DiscretizedSpaceElement` instance"
            )
        out_element = out.data
    ### Unpacking the dimension of the problem
    ndim = input_data.ndim
        
    ### Unpacking the underlying arrays
    input_data_arr, input_backend = get_array_and_backend(input_data, must_be_contiguous=True)

    if input_backend.impl != 'numpy':
        out_element = np.ascontiguousarray(input_backend.to_numpy(out_element))
        input_data_arr = np.ascontiguousarray(input_backend.to_numpy(input_data_arr))

    range_backend = lookup_array_backend(range_space.impl)
    assert input_backend == range_backend, f"The input's tensor space backend does not match the range's: {input_backend} != {range_backend}"    

    # Create astra geometries
    # The volume geometry is defined by the space of the input data in the forward mode and the range_space in the backward mode
    if direction == 'forward':        
        vol_geom = astra_volume_geometry(input_data.space, 'cpu')
    else:
        vol_geom = astra_volume_geometry(range_space, 'cpu')

    # Parsing the pprojection geometry does not depend on the mode       
    proj_geom = astra_projection_geometry(geometry, 'cpu')

    # Create projector
    if astra_proj_type is None:
        astra_proj_type = default_astra_proj_type(geometry)
    proj_id = astra_projector(astra_proj_type, vol_geom, proj_geom, ndim)

    # Create ASTRA data structures
    # In the forward mode, the input is the volume
    # In the backward mode, the input is the sinogram/projection
    if direction == 'forward': 
        input_id = astra_data(vol_geom, datatype='volume', data=input_data_arr,
                        allow_copy=True)
    else:
        input_id = astra_data(proj_geom, datatype='projection', data=input_data_arr, allow_copy=True
    )
    
    with writable_array(out_element, must_be_contiguous=True) as out_arr:
        if direction == 'forward': 
            output_id = astra_data(
                proj_geom, 
                datatype='projection', 
                data=out_arr,
                ndim=range_space.ndim)
            vol_id  = input_id
            sino_id = output_id
        else:
            output_id = astra_data(
                vol_geom, 
                datatype='volume', 
                data=out_arr,
                ndim=range_space.ndim)
            vol_id = output_id
            sino_id = input_id

        # Create algorithm
        algo_id = astra_algorithm(
            direction=direction,
            ndim = ndim,
            vol_id  = vol_id,
            sino_id = sino_id,
            proj_id = proj_id,
            astra_impl='cpu')

        # Run algorithm
        astra.algorithm.run(algo_id)

    # There is no scaling for the forward mode
    if direction == 'backward':
        # Weight the adjoint by appropriate weights
        scaling_factor = float(input_data.space.weighting.const)
        scaling_factor /= float(range_space.weighting.const)

        out_element *= scaling_factor

    # Delete ASTRA objects
    astra.algorithm.delete(algo_id)
    astra.data2d.delete((vol_id, sino_id))
    astra.projector.delete(proj_id)

    if out is None:
        return range_space.element(out_element)
    else:
        out.data[:] = range_space.element(out_element).data
        
class AstraCpuImpl:
    """Thin wrapper implementing ASTRA CPU for `RayTransform`."""

    def __init__(self, geometry, vol_space, proj_space):
        """Initialize a new instance.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup.
        vol_space : `DiscretizedSpace`
            Reconstruction space, the space of the images to be forward
            projected.
        proj_space : `DiscretizedSpace`
            Projection space, the space of the result.
        """
        if not isinstance(geometry, Geometry):
            raise TypeError(f"`geometry` must be a `Geometry` instance, got {geometry}")
        if not isinstance(vol_space, DiscretizedSpace):
            raise TypeError(
                f"`vol_space` must be a `DiscretizedSpace` instance, got {vol_space}"
            )
        if not isinstance(proj_space, DiscretizedSpace):
            raise TypeError(
                f"`proj_space` must be a `DiscretizedSpace` instance, got {proj_space}"
            )
        if geometry.ndim > 2:
            raise ValueError(f"`impl` {self.__class__.__name__} only works for 2d")

        if vol_space.size >= 512**2:
            warnings.warn(
                "The 'astra_cpu' backend may be too slow for volumes of this "
                "size. Consider using 'astra_cuda' if your machine has an "
                "Nvidia GPU.",
                RuntimeWarning,
            )

        self.geometry = geometry
        self._vol_space = vol_space
        self._proj_space = proj_space

    @property
    def vol_space(self):
        return self._vol_space

    @property
    def proj_space(self):
        return self._proj_space

    @_add_default_complex_impl
    def call_backward(self, x, out=None, **kwargs):
        # return astra_cpu_back_projector(
        #     x, self.geometry, self.vol_space.real_space, out, **kwargs
        # )
        return astra_cpu_projector(
            'backward', x, self.geometry, self.vol_space.real_space, out, **kwargs
        )

    @_add_default_complex_impl
    def call_forward(self, x, out=None, **kwargs):
        # return astra_cpu_forward_projector(
        #     x, self.geometry, self.proj_space.real_space, out, **kwargs
        # )
        return astra_cpu_projector(
            'forward', x, self.geometry, self.proj_space.real_space, out, **kwargs
        )


if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests

    run_doctests()
