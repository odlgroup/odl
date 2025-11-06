# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""NumPy implementation of tensor spaces."""

from __future__ import absolute_import, division, print_function

from odl.core.set.space import LinearSpaceElement
from odl.core.space.base_tensors import Tensor, TensorSpace
from odl.core.util import is_numeric_dtype
from odl.core.array_API_support import ArrayBackend

import numpy as np

# Only for module availability checking
import importlib.util       
from os import path
from sys import argv

torch_module = importlib.util.find_spec("torch")
if torch_module is not None:
    import torch
    import array_api_compat.torch as xp
    PYTORCH_AVAILABLE = True
else:
    if path.basename(argv[0]) == 'pytest':
        # If running the doctest suite, we should be able to load this
        # module (without running anything) even if Torch is not installed.
        PYTORCH_AVAILABLE = False
        import pytest
        pytest.skip(allow_module_level=True)
    else:
        raise ImportError("You are trying to use the PyTorch backend, but"
                        + " the `torch` dependency is not available."
                        + "\nEither use a different backend, or install"
                        + " a suitable version of Torch." )

__all__ = (
    'PYTORCH_AVAILABLE',
    'PyTorchTensorSpace',
    'pytorch_array_backend'
    
    )
if PYTORCH_AVAILABLE:
    device_strings = ['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]

def to_numpy(x):
    if isinstance(x, (int, float, bool, complex)):
        return x    
    elif isinstance(x, Tensor):
        return x.data.detach().cpu().numpy()
    else:
        return x.detach().cpu().numpy()

def from_dlpack(x, device='cpu', copy=None):
    """This should theoretically be a stand-in for `from_dlpack` in the Torch instantiation
    of the Array API. That function varies however in behaviour between current PyTorch versions,
    causing numerous failures. So instead, for now we manually implement conversions from the
    alternative backends relevant to ODL (at the moment, NumPy and PyTorch itself).
    """
    if isinstance(x, torch.Tensor):
        if x.device == device and copy != True:
            return x
        # The redundant-looking copy logic is because Torch does not recognize `copy=None`,
        # aka `AVOID_UNNECESSARY_COPY`.
        return x.to(device, copy = True if copy else False)
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, device=torch.device(device))
    else:
        raise NotImplementedError(f"With PyTorch {torch.__version__}, currently no way to handle input of type {type(x)}.")

if PYTORCH_AVAILABLE:
  pytorch_array_backend = ArrayBackend(
    impl = 'pytorch',
    available_dtypes = {      
        "bool" : xp.bool,
        "int8" : xp.int8,
        "int16" : xp.int16,
        "int32" : xp.int32,
        "int64" : xp.int64,
        "uint8" : xp.uint8,
        "uint16" : xp.uint16,
        "uint32" : xp.uint32,
        "uint64" : xp.uint64,
        "float32" : xp.float32,
        "float64" :xp.float64,
        "complex64" : xp.complex64,
        "complex128" : xp.complex128,
      },
    array_namespace = xp,
    array_constructor = xp.asarray,
    from_dlpack = from_dlpack,
    array_type = xp.Tensor,
    make_contiguous = lambda x: x if x.data.is_contiguous() else x.contiguous(),
    identifier_of_dtype = lambda dt: (dt) if dt in [int, bool, float, complex] else str(dt).split('.')[-1], 
    available_devices = device_strings,
    to_cpu = lambda x: x if isinstance(x, (int, float, bool, complex)) else x.detach().cpu(),
    to_numpy = to_numpy,
    to_device = lambda x, device: x.to(device)
   )
else:
    pytorch_array_backend = None

class PyTorchTensorSpace(TensorSpace):

    """Set of tensors of arbitrary data type, implemented with PyTorch.

    A tensor is, in the most general sense, a multi-dimensional array
    that allows operations per entry (keep the rank constant),
    reductions / contractions (reduce the rank) and broadcasting
    (raises the rank).
    For non-numeric data type like ``object``, the range of valid
    operations is rather limited since such a set of tensors does not
    define a vector space.
    Any numeric data type, on the other hand, is considered valid for
    a tensor space, although certain operations - like division with
    integer dtype - are not guaranteed to yield reasonable results.

    Under these restrictions, all basic vector space operations are
    supported by this class, along with reductions based on arithmetic
    or comparison, and element-wise mathematical functions.

    This class is implemented using `torch.Tensor`'s as back-end.

    See the `Wikipedia article on tensors`_ for further details.
    See also [Hac2012] "Part I Algebraic Tensors" for a rigorous
    treatment of tensors with a definition close to this one.

    Note also that this notion of tensors is the same as in popular
    Deep Learning frameworks.

    References
    ----------
    [Hac2012] Hackbusch, W. *Tensor Spaces and Numerical Tensor Calculus*.
    Springer, 2012.

    .. _Wikipedia article on tensors: https://en.wikipedia.org/wiki/Tensor
    """

    def __init__(self, shape, dtype='float64', device = 'cpu', requires_grad=False, **kwargs):
        r"""Initialize a new instance.

        Parameters
        ----------
        shape : positive int or sequence of positive ints
            Number of entries per axis for elements in this space. A
            single integer results in a space with rank 1, i.e., 1 axis.
        dtype (str): optional
            Data type of each element. Defaults to 'float64'
        device (str):
            Device on which the data is. Defaults to 'cpu'
        requires_grad (bool):
            Is True if gradients need to be computed for this Tensor using PyTorch's autograd engine, False otherwise.

        Other Parameters
        ----------------
        weighting : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weighting``:

            ``None``: no weighting, i.e. weighting with ``1.0`` (default).

            `Weighting`: Use this weighting as-is. Compatibility
            with this space's elements is not checked during init.

            ``float``: Weighting by a constant.

            array-like: Pointwise weighting by an array.

            This option cannot be combined with ``dist``,
            ``norm`` or ``inner``. It also cannot be used in case of
            non-numeric ``dtype``.

        dist : callable, optional
            Distance function defining a metric on the space.
            It must accept two `PyTorchTensor` arguments and return
            a non-negative real number. See ``Notes`` for
            mathematical requirements.

            By default, ``dist(x, y)`` is calculated as ``norm(x - y)``.

            This option cannot be combined with ``weight``,
            ``norm`` or ``inner``. It also cannot be used in case of
            non-numeric ``dtype``.

        norm : callable, optional
            The norm implementation. It must accept a
            `PyTorchTensor` argument, return a non-negative real number.
            See ``Notes`` for mathematical requirements.

            By default, ``norm(x)`` is calculated as ``inner(x, x)``.

            This option cannot be combined with ``weight``,
            ``dist`` or ``inner``. It also cannot be used in case of
            non-numeric ``dtype``.

        inner : callable, optional
            The inner product implementation. It must accept two
            `PyTorchTensor` arguments and return an element of the field
            of the space (usually real or complex number).
            See ``Notes`` for mathematical requirements.

            This option cannot be combined with ``weight``,
            ``dist`` or ``norm``. It also cannot be used in case of
            non-numeric ``dtype``.

        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no
            inner product is defined.

            This option has no impact if either ``dist``, ``norm`` or
            ``inner`` is given, or if ``dtype`` is non-numeric.

            Default: 2.0

        kwargs :
            Further keyword arguments are passed to the weighting
            classes.

        See Also
        --------
        odl.core.space.space_utils.rn : constructor for real tensor spaces
        odl.core.space.space_utils.cn : constructor for complex tensor spaces
        odl.core.space.space_utils.tensor_space :
            constructor for tensor spaces of arbitrary scalar data type

        Notes
        -----
        - A distance function or metric on a space :math:`\mathcal{X}`
          is a mapping
          :math:`d:\mathcal{X} \times \mathcal{X} \to \mathbb{R}`
          satisfying the following conditions for all space elements
          :math:`x, y, z`:

          * :math:`d(x, y) \geq 0`,
          * :math:`d(x, y) = 0 \Leftrightarrow x = y`,
          * :math:`d(x, y) = d(y, x)`,
          * :math:`d(x, y) \leq d(x, z) + d(z, y)`.

        - A norm on a space :math:`\mathcal{X}` is a mapping
          :math:`\| \cdot \|:\mathcal{X} \to \mathbb{R}`
          satisfying the following conditions for all
          space elements :math:`x, y`: and scalars :math:`s`:

          * :math:`\| x\| \geq 0`,
          * :math:`\| x\| = 0 \Leftrightarrow x = 0`,
          * :math:`\| sx\| = |s| \cdot \| x \|`,
          * :math:`\| x+y\| \leq \| x\| +
            \| y\|`.

        - An inner product on a space :math:`\mathcal{X}` over a field
          :math:`\mathbb{F} = \mathbb{R}` or :math:`\mathbb{C}` is a
          mapping
          :math:`\langle\cdot, \cdot\rangle: \mathcal{X} \times
          \mathcal{X} \to \mathbb{F}`
          satisfying the following conditions for all
          space elements :math:`x, y, z`: and scalars :math:`s`:

          * :math:`\langle x, y\rangle =
            \overline{\langle y, x\rangle}`,
          * :math:`\langle sx + y, z\rangle = s \langle x, z\rangle +
            \langle y, z\rangle`,
          * :math:`\langle x, x\rangle = 0 \Leftrightarrow x = 0`.

        Examples
        --------
        Explicit initialization with the class constructor:

        >>> space = PyTorchTensorSpace(3, dtype=float)
        >>> space
        rn(3, 'float64', 'pytorch')
        >>> space.shape
        (3,)
        >>> space.dtype
        torch.float64
        """
        super(PyTorchTensorSpace, self).__init__(shape, dtype, device, **kwargs)

    ########## Attributes ##########
    @property
    def array_backend(self) -> ArrayBackend:
        return pytorch_array_backend
    
    @property
    def array_namespace(self):
        """Name of the array_namespace"""
        return xp
    
    @property
    def element_type(self):
        """Type of elements in this space: `PyTorchTensor`."""
        return PyTorchTensor
    
    @property
    def impl(self):
        """Name of the implementation back-end: ``'pytorch'``."""
        return 'pytorch'

    ######### public methods #########
    def broadcast_to(self, inp):
        arr = self.array_namespace.broadcast_to(
                    self.array_namespace.asarray(inp, device=self.device),
                    self.shape
                    )
        return arr

    ######### private methods #########    

class PyTorchTensor(Tensor):

    """Representation of a `PyTorchTensorSpace` element.

    This is an internal ODL class; it should not directly be instantiated from user code.
    Instead, always use the `.element` method of the `space` for generating these objects."""
    
    def __init__(self, space, data):
        """Initialize a new instance."""
        # Tensor.__init__(self, space)
        LinearSpaceElement.__init__(self, space)
        assert(isinstance(data, xp.Tensor)), f"{type(data)=}, should be torch.Tensor"
        if data.dtype != space.dtype:
            data = data.to(space.dtype)
        if data.device != space.device:
            data = data.to(space.device)
        self.__data = data

    @property
    def data(self):
        """The `torch.Tensor` representing the data of ``self``."""
        return self.__data
    
    @data.setter
    def data(self, value):
        self.__data = value 
    
    
    def _assign(self, other, avoid_deep_copy):
        """Assign the values of ``other``, which is assumed to be in the
        same space, to ``self``."""
        if avoid_deep_copy:
            self.__data = other.__data
        else:
            self.__data[:] = other.__data

    ######### Public methods #########        
    def copy(self):
        """Return an identical (deep) copy of this tensor.

        Parameters
        ----------
        None

        Returns
        -------
        copy : `PyTorchTensor`
            The deep copy

        Examples
        --------
        >>> space = odl.rn(3, impl='pytorch')
        >>> x = space.element([1, 2, 3])
        >>> y = x.copy()
        >>> y == x
        True
        >>> y is x
        False
        """
        return self.space.element(self.data.clone())

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be accessed.

        Returns
        -------
        values : `PyTorchTensorSpace.dtype` or `PyTorchTensor`
            The value(s) at the given indices. Note that the returned
            object is a writable view into the original tensor, except
            for the case when ``indices`` is a list.

        Examples
        --------
        For one-dimensional spaces, indexing is as in linear arrays:

        >>> space = odl.rn(3, impl='pytorch')
        >>> x = space.element([1, 2, 3])
        >>> x[0]
        1.0
        >>> x[1:]
        rn(2, 'float64', 'pytorch').element([ 2.,  3.])

        In higher dimensions, the i-th index expression accesses the
        i-th axis:

        >>> space = odl.rn((2, 3), impl='pytorch')
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x[0, 1]
        2.0
        >>> x[:, 1:]
        rn((2, 2), 'float64', 'pytorch').element(
            [[ 2.,  3.],
             [ 5.,  6.]]
        )

        Slices can be assigned to, except if lists are used for indexing:

        >>> y = x[:, ::2]  # view into x
        >>> y[:] = -9
        >>> x
        rn((2, 3), 'float64', 'pytorch').element(
            [[-9.,  2., -9.],
             [-9.,  5., -9.]]
        )
        >>> y = x[[0, 1], [1, 2]]  # not a view, won't modify x
        >>> y
        rn(2, 'float64', 'pytorch').element([ 2., -9.])
        >>> y[:] = 0
        >>> x
        rn((2, 3), 'float64', 'pytorch').element(
            [[-9.,  2., -9.],
             [-9.,  5., -9.]]
        )
        """
        # Lazy implementation: index the array and deal with it
        if isinstance(indices, type(self)):
            indices = indices.data
        arr = self.data[indices]

        if arr.ndim == 0:
            if self.space.field is not None:
                return self.space.field.element(arr)
            else:
                return arr
        else:
            if is_numeric_dtype(self.dtype):
                weighting = self.space.weighting
            else:
                weighting = None
            space = type(self.space)(
                arr.shape, dtype=self.dtype, exponent=self.space.exponent,
                weighting=weighting, device=self.device)
            return space.element(arr, copy=False)

    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be written to.
        values : scalar, array-like or `PyTorchTensor`
            The value(s) that are to be assigned.

            If ``index`` is an integer, ``value`` must be a scalar.

            If ``index`` is a slice or a sequence of slices, ``value``
            must be broadcastable to the shape of the slice.

        Examples
        --------
        For 1d spaces, entries can be set with scalars or sequences of
        correct shape:

        >>> space = PyTorchTensorSpace(3, dtype=float)
        >>> x = space.element([1, 2, 3])
        >>> x[0] = -1
        >>> x[1:] = (0, 1)
        >>> x
        rn(3, 'float64', 'pytorch').element([-1.,  0.,  1.])

        It is also possible to use tensors of other spaces for
        casting and assignment:

        >>> space = PyTorchTensorSpace((2,3), dtype=float)
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x[0, 1] = -1
        >>> x
        rn((2, 3), 'float64', 'pytorch').element(
            [[ 1., -1.,  3.],
             [ 4.,  5.,  6.]]
        )
        >>> short_space = PyTorchTensorSpace((2, 2), dtype='int32')
        >>> y = short_space.element([[-1, 2],
        ...                          [0, 0]])
        >>> x[:, :2] = y
        >>> x
        rn((2, 3), 'float64', 'pytorch').element(
            [[-1.,  2.,  3.],
             [ 0.,  0.,  6.]]
        )

        The PyTorch assignment and broadcasting rules apply:

        >>> x[:] = torch.tensor([[0, 0, 0],
        ...                  [1, 1, 1]])
        >>> x
        rn((2, 3), 'float64', 'pytorch').element(
            [[ 0.,  0.,  0.],
             [ 1.,  1.,  1.]]
        )
        >>> x[:, 1:] = [7, 8]
        >>> x
        rn((2, 3), 'float64', 'pytorch').element(
            [[ 0.,  7.,  8.],
             [ 1.,  7.,  8.]]
        )
        >>> x[:, ::2] = -2.
        >>> x
        rn((2, 3), 'float64', 'pytorch').element(
            [[-2.,  7., -2.],
             [-2.,  7., -2.]]
        )
        """
        if isinstance(indices, type(self)):
            indices = indices.data
        if isinstance(values, type(self)):
            values = values.data

        if isinstance(values, (list, tuple)):
            values = self.array_backend.array_constructor(values)
        self.data[indices] = values

if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests
    run_doctests()
