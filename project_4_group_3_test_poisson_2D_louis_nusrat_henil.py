import pytest
import numpy as np
from poisson import create_laplacian_2d

def test_laplacian_2d():
    laplacian = create_laplacian_2d(3, 1, pbc=True)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-4.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [1.0, -4.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                  [1.0, 1.0, -4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, -4.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0, 1.0, -4.0, 0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -4.0, 1.0, 1.0],
                  [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -4.0, 1.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -4.0]]) * (3 / 1)**2)
    laplacian = create_laplacian_2d(3, 1, pbc=False)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-4.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [1.0, -4.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, -4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, -4.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 1.0, -4.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -4.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -4.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -4.0]]) * (3 / 1)**2)

@pytest.mark.parametrize('n_boxes,box_length,pbc,exception', [
    (1, 1, True, ValueError),
    (3, -1, True, ValueError),
    ('hello', 1, True, TypeError),
    (3, None, True, TypeError),
    (3, 1, 'hello', TypeError)])
def test_laplacian_2d_exceptions(n_boxes, box_length, pbc, exception):
    with pytest.raises(exception):
        create_laplacian_2d(n_boxes, box_length, pbc=pbc)




