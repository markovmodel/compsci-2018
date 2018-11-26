import numpy as np


def create_laplacian_2d(nx, ny, lx, ly, pbc=True):
    """ Computes discrete Laplacian for a 2d
        charge density matrix, ordered row-wise
        Args:
            nx: number of grid points along x axis, nx >= 2
            ny: number of grid points along y axis, ny >= 2
            Lx: length of grid along x axis, Lx > 0
            Ly: length of grid along y axis, Ly > 0
            pbc: periodic boundry conditions, boolean
        output:
            Laplacian as nx * ny by nx * ny np.array
    """
    if type(nx) != int or type(ny) != int:
        raise TypeError('We need an integer')
    if type(lx) != int and type(lx) != float:
        raise TypeError('We need a number')
    if type(ly) != int and type(ly) != float:
        raise TypeError('We need a number')
    if nx < 2 or ny < 2:
        raise ValueError('We need at least two grid points')
    if lx <= 0 or ly <= 0:
        raise ValueError('We need a positive length')
    if type(pbc) != bool:
        raise TypeError('We need a boolean as pbc')

    hx = (nx / lx) ** 2
    hy = (ny / ly) ** 2
    a1 = (-2 * hx - 2 * hy) * np.diag(np.ones(nx * ny))
    a2 = np.diag([0 if i % nx == 0 else hx for i in range(1, nx * ny)], 1)
    a3 = np.diag([0 if i % nx == 0 else hx for i in range(1, nx * ny)], -1)
    a4 = hy * np.diag(np.ones(nx * ny - nx), nx)
    a5 = hy * np.diag(np.ones(nx * ny - nx), -nx)
    laplacian = a1 + a2 + a3 + a4 + a5

    if pbc:
        a6 = hy * np.diag(np.ones(nx), nx * ny - nx)
        a7 = hy * np.diag(np.ones(nx), -nx * ny + nx)
        a8 = np.diag([hx if i % nx == 0 else 0
                     for i in range(0, nx * ny - nx + 1)], nx - 1)
        a9 = np.diag([hx if i % nx == 0 else 0
                     for i in range(0, nx * ny - nx + 1)], -nx + 1)
        laplacian += a6 + a7 + a8 + a9

    return laplacian
