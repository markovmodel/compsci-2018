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

def findIndex_(x, y, nx, ny):
    """Finds the actually index in the 2-D Laplacian based on x, y coordinates.
    """
    x, y = x % nx, y % ny
    return x + y * nx

def create_laplacian_2d(nx, ny, lx, ly, pbc = True):
    """Creates a discretized Laplacian in 2D

    Arguments:
        nx, ny (int): number of grid points of x and y directions; needs more than one
        lx, ly (float): box lenght along x and y; must be positive
        pbc (boolean): use periodic boundary conditions
    """
    if(nx < 2 or ny < 2):
        raise ValueError('We need at least two grid points')
    if(lx <= 0.0 or ly <= 0.0):
        raise ValueError('We need a positive length')
    if pbc not in (True, False):
        raise TypeError('We need a boolean as pbc')
    laplacian = np.zeros((nx * ny, nx * ny))
    mx = (nx / lx)**2
    my = (ny / ly)**2
    mxy = -2*(nx / lx)**2 - 2*(ny / ly)**2
    findIndex = lambda x, y: findIndex_(x, y, nx, ny)
    for x in range(nx):
        for y in range(ny):
            laplacian[findIndex(x, y), findIndex(x, y)] += mxy
            laplacian[findIndex(x, y), findIndex(x - 1, y)] += mx
            laplacian[findIndex(x, y), findIndex(x + 1, y)] += mx
            laplacian[findIndex(x, y), findIndex(x, y - 1)] += my
            laplacian[findIndex(x, y), findIndex(x, y + 1)] += my
    if not pbc:
        # modified for boundary condition calc. 
        # Should not just be set to 0, or it will be incorrect for 2x2 cases
        for y in range(ny):
            laplacian[findIndex(0, y), findIndex(-1, y)] -= mx
            laplacian[findIndex(nx - 1, y), findIndex(0, y)] -= mx
        for x in range(nx):
            laplacian[findIndex(x, 0), findIndex(x, -1)] -= my
            laplacian[findIndex(x, ny - 1), findIndex(x, 0)] -= my
    return laplacian
