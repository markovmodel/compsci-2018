import pytest
import numpy as np
from .poisson_2 import create_laplacian_2d


### test laplacian 3 * 3, hardcoded ###
def test_laplacian_2d():
    laplacian = create_laplacian_2d(3, 3, 1, 1, pbc=True)
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
    laplacian = create_laplacian_2d(3, 3, 1, 1, pbc=False)
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

### test ValueError ###
@pytest.mark.parametrize('nx,ny,lx,ly,pbc,exception', [
    (1, 3, 1, 1, True, ValueError),
    (3, 1, 1, 1, True, ValueError),
    (3, 3, 1, -1, True, ValueError),
    (3, 3, -1, 1, True, ValueError),
    ])
def test_laplacian_2d_exceptions(nx, ny, lx, ly, pbc, exception):
    with pytest.raises(exception):
        create_laplacian_2d(nx, ny, lx, ly, pbc=pbc)

### test TypeError ###
list1 = [(i, 3, 1, 1, True, TypeError)
     for i in [[0], 'string', {'dict': 0}, True]]
list1.extend([(3, i, 1, 1, True, TypeError)
    for i in [[0], 'string', {'dict': 0}, True]])
list1.extend([(3, 3, i, 1, True, TypeError)
    for i in [[0], 'string', {'dict': 0}, True]])
list1.extend([(3, 3, 1, i, True, TypeError)
    for i in [[0], 'string', {'dict': 0}, True]])
list1.extend([(3, 3, 1, 1, i, TypeError)
    for i in [[0], 'string', {'dict': 0}]])
@pytest.mark.parametrize('nx, ny, lx, ly,pbc,exception', list1)
def test_laplacian_2d_exceptions(nx, ny, lx, ly, pbc, exception):
    with pytest.raises(exception):
        create_laplacian_2d(nx, ny, lx, ly, pbc=pbc)

### test periodc grid ###
@pytest.mark.parametrize('nx, ny , lx, ly', [
    (3, 3, 2, 2),
    (3, 3, 1, 1),
    (2, 2, 1, 1),
    (2, 3, 1, 1),
    (3, 2, 1, 1),
    ])
def test_laplacian_2d(nx, ny, lx, ly):
    x = np.linspace(0, lx * 2 * np.pi, nx)
    y = np.linspace(0, ly * 2 * np.pi, ny)
    xx, yy = np.meshgrid(x,y)
    grid = np.sin(xx) * np.cos(yy)
    laplacian = create_laplacian_2d(nx, ny, lx * 2 * np.pi, ly * 2 * np.pi)
    grid_lap = np.dot(laplacian, grid.reshape(nx * ny, 1))
    np.testing.assert_almost_equal(grid, grid_lap.reshape(ny, nx))
