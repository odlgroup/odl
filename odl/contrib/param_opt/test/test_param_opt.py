import pytest
import odl
import odl.contrib.fom
import odl.contrib.param_opt
from odl.util.testutils import simple_fixture

space = simple_fixture('space',
                       [odl.rn(3),
                        odl.uniform_discr([0, 0], [1, 1], [9, 11]),
                        odl.uniform_discr(0, 1, 10)])

def test_optimal_parameters(space):
    """Tests if optimal_parameters works for some simple examples."""
    fom = odl.contrib.fom.mean_squared_error
    mynoise = odl.phantom.white_noise(space)
    phantoms = [mynoise]
    data = [mynoise]

    def reconstruction(data, lam):
        """Perturbs the data by adding lam to it."""
        return data + lam

    result = odl.contrib.param_opt.optimal_parameters(reconstruction, fom,
                                                      phantoms, data, 1)
    assert result == pytest.approx(0)


if __name__ == '__main__':
    odl.util.test_file(__file__)
