import numpy as np

def DiffusionKernel(mat):
    """Applies diffusion operation on a discrete grid (lattice)
    with periodic boundaries (this is called discrete LaPlacian)
    credit: Ben Maier
    """
    #Input: 2D numpy array
    #Output: 2D numpy array 

    # the cell appears 4 times in the formula to compute
    # # the total difference
    neigh_mat = -4*mat.copy()
    # Each direct neighbor on the lattice is counted in
    # # the discrete difference formula
    neighbors = [( 1.0,  (-1, 0) ),( 1.0,  ( 0,-1) ),( 1.0,  ( 0, 1) ),( 1.0,  ( 1, 0) ),]
    # shift matrix according to demanded neighbors
    # # and add to this cell with corresponding weight
    # for weight, neigh in neighbors:
    neigh_mat += weight * np.roll(mat, neigh, (0,1))
    return neigh_mat