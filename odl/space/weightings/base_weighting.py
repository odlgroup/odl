import odl       

from odl.util import signature_string, array_str, indent

def not_implemented(
        *args        
        ):
    raise NotImplementedError

class Weighting(object):
    def __init__(self, device, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        
        """        
        self._inner = self._inner_default
        self._array_norm  = self._norm_default
        self._dist  = self._dist_default
        self._exponent = 2.0
        self._weight = 1.0
        self._shape = None

        # Check device consistency and allocate __device attribute
        self.parse_device(device)
        # Overload of the default attributes and methods if they are found in the kwargs
        self.parse_kwargs(kwargs)

    def parse_device(self, device):
        # Checks
        odl.check_device(self.impl, device)
        # Set attribute   
        self._device = device

    def parse_kwargs(self, kwargs):
        if 'exponent' in kwargs:
            # Pop the kwarg
            exponent = kwargs.pop('exponent')
            # Check the kwarg
            if exponent <= 0:
                raise ValueError(
                    f"only positive exponents or inf supported, got {exponent}"
                    )
            # Assign the attribute
            self._exponent = exponent
            if self.exponent != 2:
                self._inner = not_implemented

        if 'inner' in kwargs:
            # Pop the kwarg
            inner = kwargs.pop('inner')
            # check the kwarg
            assert callable(inner)
            # Check the consistency
            assert self.exponent == 2.0
            assert not set(['norm', 'dist', 'weight']).issubset(kwargs)
            # Assign the attribute       
            self._inner = inner
            
        elif 'norm' in kwargs:
            # Pop the kwarg
            array_norm = kwargs.pop('norm')
            # check the kwarg
            assert callable(array_norm)
            # Check the consistency
            assert self.exponent == 2.0
            assert not set(['inner', 'dist', 'weight']).issubset(kwargs)
            # Assign the attributes
            self._inner = not_implemented
            self._array_norm  = array_norm
        
        elif 'dist' in kwargs:
            # Pop the kwarg
            dist  = kwargs.pop('dist')
            # check the kwarg
            assert callable(dist)
            # Check the consistency
            assert self.exponent == 2.0
            assert not set(['inner', 'norm', 'weight']).issubset(kwargs)
            # Assign the attributes
            self._inner = not_implemented
            self._array_norm  = not_implemented
            self._dist  = dist
        
        elif 'weight' in kwargs:
            # Pop the kwarg
            weight = kwargs.pop('weight')
            # Check the consistency
            assert not set(['inner', 'norm', 'dist']).issubset(kwargs)
            # check the kwarg AND assign the attribute
            if isinstance(weight, (int, float)):
                if 0 < weight and weight != float('inf'):
                    self._weight = float(weight)
                else:
                    raise ValueError("If the weight if a float, it must be positive")
            
            elif hasattr(weight, 'odl_tensor'):
                if self.array_namespace.all(0 < weight.data):
                    self._weight = weight.data
                    self._shape = self.weight.shape
                    assert self.impl == weight.impl
                    assert self.device == weight.device
                else:
                    raise ValueError("If the weight if an ODL Tensor, all its entries must be positive")
                
            elif hasattr(weight, '__array__'):
                if self.array_namespace.all(0 < weight):
                    self._weight = weight
                    self._shape = self.weight.shape
                    assert isinstance(self.weight, self.array_type)
                    assert self.device == weight.device
                else:
                    raise ValueError("If the weight if an array, all its elements must be positive")          

            else:
                raise ValueError(f"A weight can only be a positive __array__, a positive float or a positive ODL Tensor")      

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unknown keyword arguments {}'.format(kwargs))

    @property
    def device(self):
        """Device of this weighting."""
        return self._device
    
    @property
    def exponent(self):
        """Exponent of this weighting."""
        return self._exponent
    
    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        optargs = [('weight', array_str(self.weight), array_str(1.0)),
                   ('exponent', self.exponent, 2.0),
                   ('inner', self._inner, self._inner_default),
                   ('norm', self._array_norm, self._norm_default),
                   ('dist', self._dist, self._norm_default),
                   ]
        return signature_string([], optargs, sep=',\n',
            mod=[[], ['!s', ':.4', '!r', '!r', '!r']])
    
    @property
    def shape(self):
        """Shape of the weighting"""
        return self._shape 
    
    @property
    def weight(self):
        """Weight of this weighting."""
        return self._weight

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : bool
            ``True`` if ``other`` is a the same weighting, ``False``
            otherwise.

        Notes
        -----
        This operation must be computationally cheap, i.e. no large
        arrays may be compared entry-wise. That is the task of the
        `equiv` method.
        """
        return (isinstance(other, Weighting) and
                self.impl == other.impl and
                self.device == other.device and
                self.exponent == other.exponent and
                self.shape == other.shape and 
                self.array_namespace.equal(self.weight, other.weight).all() and                
                self._inner.__code__ == other._inner.__code__ and 
                self._array_norm.__code__ == other._array_norm.__code__ and 
                self._dist.__code__ == other._dist.__code__
                )
    def __neq__(self, other):
        return not self.__eq__(self, other)
    
    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((
            type(self), self.impl, self.device, 
            self.weight, self.exponent,
            self._inner.__code__, self._array_norm.__code__, self._dist.__code__
            ))
    
    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('weight', array_str(self.weight), array_str(1.0)),
                   ('exponent', self.exponent, 2.0),
                   ('inner', self._inner, self._inner_default),
                   ('norm', self._array_norm, self._norm_default),
                   ('dist', self._dist, self._dist_default),
                   ]
        inner_str = signature_string([], optargs, sep=',\n',
            mod=[[], ['!s', ':.4', '!r', '!r', '!r']])
        return '{}(\n{}\n)'.format(self.__class__.__name__, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)
    
    def equiv(self, other):
        """Test if ``other`` is an equivalent weighting.

        Should be overridden, default tests for equality.

        Returns
        -------
        equivalent : bool
            ``True`` if ``other`` is a `Weighting` instance which
            yields the same result as this inner product for any
            input, ``False`` otherwise.
        """
        return self == other
    
    def inner(self, x1, x2):
        """Return the inner product of two elements.

        Parameters
        ----------
        x1, x2 : `LinearSpaceElement`
            Elements whose inner product is calculated.

        Returns
        -------
        inner : float or complex
            The inner product of the two provided elements.
        """
        if isinstance(self.weight, (int, float)):
            return self.weight * self._inner(x1.data, x2.data)
        
        elif isinstance(self.weight, self.array_type):
            return self._inner(x1.data*self.weight, x2.data)
        
        else:
            raise ValueError(f"The weight can only be an int, a float, or a {self.array_type}, but {type(self.weight)} was provided")

    def norm(self, x):
        """Calculate the norm of an element.

        This is the standard implementation using `inner`.
        Subclasses should override it for optimization purposes.

        Parameters
        ----------
        x1 : `LinearSpaceElement`
            Element whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the element.
        """
        return self._array_norm(x)

    def dist(self, x1, x2):
        """Calculate the distance between two elements.

        This is the standard implementation using `norm`.
        Subclasses should override it for optimization purposes.

        Parameters
        ----------
        x1, x2 : `LinearSpaceElement`
            Elements whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the elements.
        """
        return self._dist(x1, x2)
    
    def equiv(self, other):
        return (isinstance(other, Weighting) and
                self.impl == other.impl and
                self.device == other.device and                
                self.exponent == other.exponent and
                self._inner.__code__ == other._inner.__code__ and 
                self._array_norm.__code__ == other._array_norm.__code__ and 
                self._dist.__code__ == other._dist.__code__ and
                self.array_namespace.all(self.weight == other.weight)
                )
