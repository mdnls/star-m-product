'''
Goal: plot 2d plots of sum norm with respect to coordinates of M for each coordinate

Goal: plot 3d plots of sum norm with respect to pairs of coordinates of M

1. Helper to compute nuclear norm
2. Use numpy meshgrid
3. Helper: vector function (a, b) -> make params out of them

meshgrid params: dx, radius

given a pair of points, get z val for many xy
    - implement this as a function of four coordinates, dx, dy, x, y
    - then np.from_pyfunc(lambda dx, dy: f(dx, dy, fixed_x, fixed_y)) and apply this to a meshgrid of perturbations
'''

import numpy as np
import matplotlib.pyplot as plt
import tensors as T
from tensors import xM, m_SVD, normN, haarmtx

class NuclearNormLandscape():
    def __init__(self, A, M):
        '''j
        Analyze the landscape of the sum norm of A with respect to coordinate perturbations of M
        '''
        self.A = A
        self.M = M

    def nuclear_norm(self, x_, i, j):
        '''
        Compute the true nuclear norm of A with respect to a perturbed M.


        Arguments:
            x_: perturbation
            i: row index
            j: column index

        Returns:
            np.float, nuclear norm of A with respect to perturbed M. If perturbed M is not invertible, returns -1.
        '''
        M_ = np.copy(self.M)
        M_[i, j] = M_[i, j] + x_
        try:
            return normN(self.A, M_)
        except AssertionError:
            return -1

    def loss_objective(self, x_, i, j):
        '''
        Compute the product sum(U^H *M' A *M' V) where U, S, V are the *M-SVD factor matrices of A and M' is the
        perturbed M matrix.

        Arguments:
            x_: perturbation
            i: row index
            j: column index

        Returns:
            np.float, see description. If perturbed M is not invertible, returns -1.
        '''
        M_ = np.copy(self.M)
        M_[i, j] = M_[i, j] + x_
        Uhat, Shat, Vhat = m_SVD(self.A, self.M, inv_transform=False)
        Uh_hat = np.transpose(Uhat, (1, 0, 2))
        try:
            return np.sum(xM(xM(Uh_hat, self.A, M_), Vhat, M_))
        except AssertionError:
            return -1

    def plot_nnl(self, x_max, dx):
        R, C = self.M.shape
        for r in range(R):
            for c in range(C):
                xi = np.linspace(-x_max, x_max, int(2 * x_max / dx + 1))
                compute_height = np.frompyfunc(lambda dx: self.loss_objective(dx, r, c), 1, 1)
                Z = compute_height(xi)
                Z[Z == -1] = None
                print(f"Map of {r}, {c}")
                plt.subplot(R, C, C * r + c + 1)
                plt.plot(xi, Z)



if __name__ == "__main__":
    cats_R = np.load("data/Cats_RGB.npy")[:, :, :, 0]
    n = cats_R.shape[-1]
    nnl_maker = NuclearNormLandscape(cats_R, haarmtx(32))
    nnl_maker.plot_nnl(1, 0.5)
    plt.gcf().set_size_inches(n, n)
    plt.savefig("M_opt_landscape.svg")
    plt.show()
