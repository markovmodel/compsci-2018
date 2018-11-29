import numpy as np

def create_laplacian_1d(nx, lx, pbc=True):
    """Ceates a discretized Laplacian in 1D

    Arguments:
        nx (int): number of grid points; needs more than one
        lx (float): box lenght along x; must be positive
        pbc (boolean): use periodic boundary conditions
    """
    if nx < 2:
        raise ValueError('We need at least two grid points')
    if lx <= 0.0:
        raise ValueError('We need a positive length')
    if pbc not in (True, False):
        raise TypeError('We need a boolean as pbc')
    laplacian = np.zeros((nx, nx))
    mx = (nx / lx)**2
    for x in range(nx):
        laplacian[x, x] -= 2.0 * mx
        laplacian[x, (x + 1) % nx] += mx
        laplacian[(x + 1) % nx, x] += mx
    if not pbc:
        laplacian[0, -1] = 0
        laplacian[-1, 0] = 0
    return laplacian

def create_laplacian_2d(nx, ny, lx, ly, pbc=True):
    """Generates 2d Laplacian matrix with respect to grid spacing.   
       
    Creates 4d array to show components of laplace operator at each grid point
    and fills it as per discetized laplace operator taking care of
    periodic boundary conditions.  
    In the end reshapes it into 2d laplaction matrix .  
    
    Parameters:
        nx(integer): number of grid points in X-direction
        ny(integer): number of grid points in Y-direction
        hx(float): grid-spacing in X-direction
        hy(float): grid-spacing in Y-direction
    
    Returns:
        Laplacian matrix of size(nx * ny, nx * ny)  
    """
    if not pbc:
        raise NotImplementedError(
            'Nonperiodic laplacians are not yet available')
    laplacian = np.zeros(shape=(nx, ny, nx, ny), dtype=np.float64)
    mx = (nx / lx)**2
    my = (ny / ly)**2
    for x in range(nx):
        for y in range(ny):
            laplacian[x, y, x, y] = -2.0 * (mx + my)
            laplacian[x, y, (x + 1) % nx, y] += mx
            laplacian[x, y, (x - 1) % nx, y] += mx
            laplacian[x, y, x, (y + 1) % ny] += my
            laplacian[x, y, x, (y - 1) % ny] += my
    return laplacian.reshape(nx * ny, nx * ny)
