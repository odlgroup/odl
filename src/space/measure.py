from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from future.builtins import object
from future import standard_library
standard_library.install_aliases()
from abc import ABCMeta, abstractmethod, abstractproperty

class Discretization(object):
    """ A discretization of some set
    """

    __metaclass__ = ABCMeta #Set as abstract

    @abstractmethod
    def integrate(self, vector):
        """Calculate the integral of vector
        """
