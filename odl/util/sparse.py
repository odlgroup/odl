
"""
File containing a custom COO matrix representation.
In contrast to scipy the container allows for arbitrary matrix elements.
"""

__all__ = ('COOMatrix',)


class COOMatrix():
    """
    Custom COO matrix representation for creating product space operators.
    
    The columns, rows and data are stored in separate lists such that A[i[k], j[k]] = data[k].
    
    """
    def __init__(self, data, index, shape):
        
        # type check
        if len(data) != len(index[0]) or len(data) != len(index[1]):
            raise ValueError('data and index must have the same length')
        
        self.__data = data
        self.__index = index
        self.__shape = shape

    @property
    def row(self):
        return self.__index[0]
    
    @property
    def col(self):
        return self.__index[1]
    
    @property
    def shape(self):
        return self.__shape
    
    @property
    def data(self):
        return self.__data
    
    def __repr__(self):
        return f"COO matrix({self.data}, {self.index})"
