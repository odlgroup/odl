import odl       

from odl.util import signature_string, array_str, indent

def not_implemented(
        function_name:str,
        argument:str        
        ):
    raise NotImplementedError(f'The function {function_name} when the weighting was declared with {argument}.')

class Weighting(object):
    def __init__(self, device, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        
        """        
        self.__inner = self.array_namespace.inner
        self.__array_norm  = self.array_namespace.linalg.vector_norm
        self.__dist  = None
        self.__exponent = 2.0
        self.__weight = 1.0
        self.__shape = None
        self._norm_from_inner = False

        # Check device consistency and allocate __device attribute
        self.parse_device(device)
        # Overload of the default attributes and methods if they are found in the kwargs
        self.parse_kwargs(kwargs)

    def parse_device(self, device):
        # Checks
        odl.check_device(self.impl, device)
        # Set attribute   
        self.__device = device

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
            self.__exponent = exponent

        if 'inner' in kwargs:
            # Pop the kwarg
            inner = kwargs.pop('inner')
            # check the kwarg
            assert isinstance(inner, callable)
            # Check the consistency
            assert self.exponent == 2.0
            assert not set(['norm', 'dist', 'weight']).issubset(kwargs)
            # Assign the attribute       
            self.__inner = inner
            self._norm_from_inner = True
            
        elif 'norm' in kwargs:
            # Pop the kwarg
            array_norm = kwargs.pop('norm')
            # check the kwarg
            assert isinstance(array_norm, callable)
            # Check the consistency
            assert self.exponent == 2.0
            assert not set(['inner', 'dist', 'weight']).issubset(kwargs)
            # Assign the attributes
            self.__inner = not_implemented('inner', 'norm')            
            self.__array_norm  = array_norm
        
        elif 'dist' in kwargs:
            # Pop the kwarg
            dist  = kwargs.pop('dist')
            # check the kwarg
            assert isinstance(dist, callable)
            # Check the consistency
            assert self.exponent == 2.0
            assert not set(['inner', 'norm', 'weight']).issubset(kwargs)
            # Assign the attributes
            self.__inner = not_implemented('inner', 'dist')
            self.__array_norm  = not_implemented('norm', 'dist')
            self.__dist  = dist
        
        elif 'weight' in kwargs:
            # Pop the kwarg
            weight = kwargs.pop('weight')
            # Check the consistency
            assert not set(['inner', 'norm', 'dist']).issubset(kwargs)
            # check the kwarg AND assign the attribute
            if isinstance(weight, float) and (not 0 < weight):
                raise TypeError("If the weight if a float, it must be positive")
            
            elif hasattr(weight, 'odl_tensor'):
                if self.array_namespace.all(0 < weight.data):
                    self.__weight = weight.data
                    self.__shape = self.weight.shape
                    assert isinstance(self.impl, self.weight.impl)
                    assert self.device == weight.device
                else:
                    raise TypeError("If the weight if an ODL Tensor, all its entries must be positive")
                
            elif hasattr(weight, '__array__'):
                if self.array_namespace.all(0 < weight):
                    self.__weight = weight
                    self.__shape = self.weight.shape
                    assert isinstance(self.weight, self.array_type)
                    assert self.device == weight.device
                else:
                    raise TypeError("If the weight if an array, all its elements must be positive")          

        # Make sure there are no leftover kwargs
        if kwargs:
            raise TypeError('got unknown keyword arguments {}'.format(kwargs))

    @property
    def device(self):
        """Device of this weighting."""
        return self.__device
    
    @property
    def exponent(self):
        """Exponent of this weighting."""
        return self.__exponent
    
    @property
    def repr_part(self):
        """String usable in a space's ``__repr__`` method."""
        optargs = [('weight', array_str(self.weight), array_str(1.0)),
                   ('exponent', self.exponent, 2.0),
                   ('inner', self.__inner, self.array_namespace.inner),
                   ('norm', self.__array_norm, self.array_namespace.linalg.vector_norm),
                   ('dist', self.__dist, None),
                   ]
        return signature_string([], optargs, sep=',\n',
            mod=[[], ['!s', ':.4', '!r', '!r', '!r']])
    
    @property
    def shape(self):
        """Shape of the weighting"""
        return self.__shape 
    
    @property
    def weight(self):
        """Weight of this weighting."""
        return self.__weight

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
                self.array_namespace.equal(self.weight, other.weight).all() and                
                self.exponent == other.exponent and
                self.shape == other.shape and
                self.__inner == other.__inner and 
                self.__array_norm == other.__array_norm and
                self.__dist == other.__dist
                )
    
    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((
            type(self), self.impl, self.device, 
            self.weight, self.exponent, 
            self.inner, self.norm, self.dist
            ))
    
    def __repr__(self):
        """Return ``repr(self)``."""
        optargs = [('weight', array_str(self.weight), array_str(1.0)),
                   ('exponent', self.exponent, 2.0),
                   ('inner', self.__inner, self.array_namespace.inner),
                   ('norm', self.__array_norm, self.array_namespace.linalg.vector_norm),
                   ('dist', self.__dist, None),
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
        return self.__inner((self.__weight * x1.data).ravel(), x2.data.ravel())

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
        if self._norm_from_inner:
            return self.array_namespace.sqrt(self.inner(x,x))
        else:
            return self.__array_norm(self.__weight * x.data, ord=self.exponent)

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
        if self.__dist is None:
            return self.norm(x1-x2)
        else:
            return self.__dist(x1,x2)