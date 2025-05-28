import odl       

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
        # Checks
        odl.check_device(self.impl, device)
        # Set default attributes        
        self.__device = device
        self.__inner = self.array_namespace.inner
        self.__array_norm  = self.array_namespace.linalg.vector_norm
        self.__dist  = self.array_namespace.linalg.vector_norm
        self.__exponent = 2.0
        self.__weight = 1.0

        # Overload of the default attributes and methods if they are found in the kwargs
        self.parse_kwargs(kwargs)

    def parse_kwargs(self, kwargs):
        if 'exponent' in kwargs:
            exponent = kwargs['exponent'] 
            if exponent <= 0:
                raise ValueError('only positive exponents or inf supported, '
                                'got {}'.format(self.__exponent))
            self.__exponent = exponent

        if 'inner' in kwargs:
            assert self.exponent == 2.0
            assert not set(['norm', 'dist', 'weight']).issubset(kwargs)
            self.__inner = kwargs['inner']
            
        elif 'norm' in kwargs:
            assert self.exponent == 2.0
            assert not set(['inner', 'dist', 'weight']).issubset(kwargs)
            self.__inner = not_implemented('inner', 'norm')
            self.__array_norm  = kwargs['norm']
        
        elif 'dist' in kwargs:
            assert self.exponent == 2.0
            assert not set(['inner', 'norm', 'weight']).issubset(kwargs)
            self.__inner = not_implemented('inner', 'dist')
            self.__array_norm  = not_implemented('norm', 'dist')
            self.__dist  = kwargs['dist']
        
        elif 'weight' in kwargs:
            assert not set(['inner', 'norm', 'dist']).issubset(kwargs)
            weight = kwargs['weight']

            if isinstance(weight, float) and (not 0 < weight):
                raise TypeError("If the weight if a float, it must be positive")
            
            elif hasattr(weight, 'odl_tensor'):
                if self.array_namespace.all(0 < weight.data):
                    self.__weight = weight.data
                else:
                    raise TypeError("If the weight if an ODL Tensor, all its entries must be positive")
                
            elif hasattr(weight, '__array__'):
                if self.array_namespace.all(0 < weight):
                    self.__weight = weight
                else:
                    raise TypeError("If the weight if an array, all its elements must be positive")
    @property
    def device(self):
        """Device of this weighting."""
        return self.__device
    
    @property
    def exponent(self):
        """Exponent of this weighting."""
        return self.__exponent
    
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
                self.device == other.device,
                self.weight == other.weight and                
                self.exponent == other.exponent and
                self.inner == other.inner and 
                self.norm == other.norm and
                self.dist == other.dist
                )
    
    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.impl, self.weight, self.exponent))

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
        return self.__dist(x1 - x2)