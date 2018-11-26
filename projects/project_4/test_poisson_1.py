import pytest
import numpy as np
from project_4.poisson_1 import create_laplacian_1d
from project_4.poisson_1 import create_laplacian_2d


def test_laplacian_1d():
    laplacian = create_laplacian_1d(3, 1, pbc=True)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-2.0, 1.0, 1.0],
                  [1.0, -2.0, 1.0],
                  [1.0, 1.0, -2.0]]) * (3 / 1)**2)
    laplacian = create_laplacian_1d(3, 1, pbc=False)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-2.0, 1.0, 0.0],
                  [1.0, -2.0, 1.0],
                  [0.0, 1.0, -2.0]]) * (3 / 1)**2)


@pytest.mark.parametrize('nx,lx,pbc,exception', [
    (1, 1, True, ValueError),
    (3, -1, True, ValueError),
    ('hello', 1, True, TypeError),
    (3, None, True, TypeError),
    (3, 1, 'hello', TypeError)])
def test_laplacian_1d_exceptions(nx, lx, pbc, exception):
    with pytest.raises(exception):
        create_laplacian_1d(nx, lx, pbc=pbc)
        
@pytest.mark.parametrize('nx,ny,lx,ly,pbc,exception', [
    (1, 1, 1, 1, True, ValueError),
    (3, 3, -1, -1, True, ValueError),
    ('hello', 'hello', 1, 1, True, TypeError),
    (3, 3, None, None, True, TypeError),
    (3, 3, 1, 1, 'hello', TypeError)])
def test_laplacian_2d_exceptions(nx, ny, lx, ly, pbc, exception):
    with pytest.raises(exception):
        create_laplacian_2d(nx, ny, lx, ly, pbc=pbc)

def test_laplacian_2d():
    laplacian = create_laplacian_2d(3, 3, 1, 1, pbc=True)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-4., 1., 1., 1., 0., 0., 1., 0., 0.],
                  [1., -4., 1., 0., 1., 0., 0., 1., 0.],
                  [1., 1., -4., 0., 0., 1., 0., 0., 1.],
                  [1., 0., 0., -4., 1., 1., 1., 0., 0.],
                  [0., 1., 0., 1., -4., 1., 0., 1., 0.],
                  [0., 0., 1., 1., 1., -4., 0., 0., 1.],
                  [1., 0., 0., 1., 0., 0., -4., 1., 1.],
                  [0., 1., 0., 0., 1., 0., 1., -4., 1.],
                  [0., 0., 1., 0., 0., 1., 1., 1., -4.]]) * (3 / 1)**2)
    laplacian = create_laplacian_2d(3, 3, 1, 1, pbc=False)
    np.testing.assert_almost_equal(
        laplacian,
        np.array([[-4., 1., 0., 1., 0., 0., 0., 0., 0.],
                  [1., -4., 1., 0., 1., 0., 0., 0., 0.],
                  [0., 1., -4., 0., 0., 1., 0., 0., 0.],
                  [1., 0., 0., -4., 1., 0., 1., 0., 0.],
                  [0., 1., 0., 1., -4., 1., 0., 1., 0.],
                  [0., 0., 1., 0., 1., -4., 0., 0., 1.],
                  [0., 0., 0., 1., 0., 0., -4., 1., 0.],
                  [0., 0., 0., 0., 1., 0., 1., -4., 1.],
                  [0., 0., 0., 0., 0., 1., 0., 1., -4.]]) * (3 / 1)**2)

# Anothter test
def test_laplacian_2d2():
    nx, ny, lx, ly, pbc = 5, 4, 1, 1, True
    laplacian = create_laplacian_2d(nx, ny, lx, ly, pbc)
    mx = (nx / lx)**2
    my = (ny / ly)**2
    # dimension
    assert laplacian.shape == (nx*ny, nx*ny)
    # diagonal
    assert all(value == -2*(mx + my) for value in np.diag(laplacian))
    # check diag+1 == mx or 0, and check diag-1 == mx or 0
    assert all(value == mx or value == 0 for value in np.diag(laplacian, k=1))
    assert all(value == mx or value == 0 for value in np.diag(laplacian, k=-1))
    # in pbc case, check #elements == 5 in each row & column, else 2 < #elements < 6
    #if pbc == True:
    assert all(np.count_nonzero(laplacian[i,:]) == 5 for i in range(nx*ny))
    assert all(np.count_nonzero(laplacian[:,i]) == 5 for i in range(nx*ny))
    #check this, double checking.. things above
    assert all(laplacian[x + nx*y, x + nx*y] == -(2.0 * mx + 2.0 * my) and 
                laplacian[x + nx*y, (x + y*nx + 1) % nx + y*nx] == mx and 
                laplacian[(x + y*nx + 1) % nx + y*nx, x + nx*y] == mx and 
                laplacian[x + y*nx, (x + nx*(y + 1)) % (nx*ny)] == my and 
                laplacian[(x + nx*(y + 1)) % (nx*ny), x + y*nx] == my 
                for y in range(ny) for x in range(nx))
    pbc = False
    if pbc == False:
        for y in range(ny):
            laplacian[nx*y, nx*y + nx - 1] = 0
            laplacian[nx*y + nx - 1, nx*y] = 0
        for x in range(nx):
            laplacian[x, nx*ny - (nx - x)] = 0
            laplacian[nx*ny - (nx - x), x] = 0
    assert all(np.count_nonzero(laplacian[i,:]) > 2 and np.count_nonzero(laplacian[i,:]) < 6 for i in range(nx*ny))
    assert all(np.count_nonzero(laplacian[:,i]) > 2 and np.count_nonzero(laplacian[:,i]) < 6 for i in range(nx*ny))
