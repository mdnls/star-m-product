'''
Learning the right m

Setup: need to try many approaches
1. should be convenient to load images in image format
2. should take input image, output directory, and a list of experimental parameters
3. should output a subfolder of the output dir containing a
    configs.json with all configs, an npy file of relevant outputs, and a plot (?)
4. decision to make: does this script output plots or just data for plots?



argmin_{M} U^H * X * V where U^H and V need to be orthogonal and depend on M
plan: 
    1. given X, compute V, U
    2. Fix V, U as orthogonal projectors
    3. gradient descent update to M in order to min nuclear norm of U^H X V
    4. recompute U and V
    5. orthogonalize M?
5. experimental parameters:
    - nuc on diagonal? nuc everywhere? weight?
    - orthogonalize M at each step vs. enforce orthogonality penalty vs. both
    - inner loop/outer loop parameters

does tensorflow allow this? tf inner loop, numpy outer loop
'''


import numpy as np
from tensors import  x3, xFace, xM, dftmtx, haarmtx, safe_inv, m_SVD, cT, x3_inv
import unittest
from tqdm import trange
import argparse
import matplotlib.pyplot as plt

def safe_matmul_huh(*shapes):
    '''
    Given a list of tuples representing tensor dimensions, check that the first two dimensions are compatible
        for matrix multiplication, ie. (a, b, ...), (b, d, ...), (d, e, ...) are compatible
        while (a, b, ...), (d, e, ...) are not.

    Args:
        shapes (list of tuples of integers): dimensions for matrix multiplication

    Returns:
        boolean: True if dimensions are compatible, false if not
    '''
    if len(shapes) == 1:
        return True
    elif len(shapes) < 1:
        raise ValueError("safe_matmul_huh requires at least one shape")
    else:
        return (shapes[0][1] == shapes[1][0]) and safe_matmul_huh(shapes[1:])


def star_m_grad(A, B, M, m_idx):
    '''
    Given product matrices A, B, a star-M product matrix, and a matrix index (i, j),
    return the derivative of A *M B with respect to M_{i, j}

    Args:
        A: an n1 x n2 x n3 tensor
        B: an n2 x n4 x n3 tensor
        M: an n3 x n3 matrix
        m_idx: a tuple (i, j) of integers

    Returns:
        An n1 x n4 x n3 tensor whose elements are gradients of A *M B wrt M_{ij}
    '''
    assert safe_matmul_huh(A.shape, B.shape), "A and B must have compatible dimensions for A *M B"

    j, i = m_idx
    Minv = safe_inv(M)

    def outer(A, B):
        return np.tensordot(A[..., None], B, (-1, -1))

    from_prods = outer( A[:, :, i] @ x3(B, M)[:, :, j], Minv[:, j:j+1] ) \
              + outer( x3(A, M)[:, :, j] @ B[:, :, i], Minv[:, j:j+1] )
    from_inv = outer( xM(A, B, M)[:, :, i], Minv[:, j:j+1])
    return from_prods - from_inv

def three_star_m_grad(A, B, C, M, m_idx):
    '''
    Given product matrices A, B, C, a star-M product matrix, and a matrix index (i, j),
    return the derivative of A *M B *M C with respect to M_{i, j}

    Args:
        A: an n1 x n2 x n3 tensor
        B: an n2 x n4 x n3 tensor
        C: an n4 x n5 x n3 tensor
        M: an n3 x n3 matrix
        m_idx: a tuple (i, j) of integers

    Returns:
        An n1 x n5 x n3 tensor whose elements are gradients of A *M B *M C wrt M_{ij}
    '''
    assert safe_matmul_huh(A.shape, B.shape, C.shape), "A, B, C must have compatible dimensions for A *M B *M C"

    j, i = m_idx
    Minv = safe_inv(M)

    def outer(A, B):
        return np.tensordot(A[..., None], B, (-1, -1))

    from_prods = outer( A[:, :, i] @ x3(B, M)[:, :, j] @ x3(C, M)[:, :, j], Minv[:, j:j+1] ) \
                 + outer( x3(A, M)[:, :, j] @ B[:, :, i] @ x3(C, M)[:, :, j], Minv[:, j:j+1] ) \
                 + outer( x3(A, M)[:, :, j] @ x3(B, M)[:, :, j] @ C[:, :, i], Minv[:, j:j+1])
    from_inv = outer( xM(xM(A, B, M), C, M)[:, :, i], Minv[:, j:j+1])
    return from_prods - from_inv

def sum_norm_m_grad(A, B, C, M, m_idx):
    '''
    Given product matrices A, B, C, a star-M product matrix, and a matrix index (i, j),
    return the derivative of |A *M B *M C|_1 with respect to M_{i, j}

    Args:
        A: an n1 x n2 x n3 tensor
        B: an n2 x n4 x n3 tensor
        C: an n4 x n5 x n3 tensor
        M: an n3 x n3 matrix
        m_idx: a tuple (i, j) of integers

    Returns:
        An n3 x n3 tensor whose elements are gradients of |A *M B *M C|_1 wrt M_{ij}
    '''
    # apply the chain rule: d/dx |f(x)|_1 = sign(f(x)) f'(x)
    return np.sum(np.sign(xM(xM(A, B, M), C, M)) * three_star_m_grad(A, B, C, M, m_idx))

def nonortho_penalty_grad(M):
    '''
    Given a square matrix M with side length k, return the gradient of the quantity

    \begin{equation}
    \left\lVert \frac{M^T M}{\|M^T M\|_F} - \frac{I}{\sqrt{k}} \right\rVert_F^2
    \end{equation}

    Args:
        M: a k x k square matrix

    Returns:
        A k x k square matrix
    '''
    k = M.shape[0]
    mtm = M.T @ M
    frobenius_mtm = np.linalg.norm(mtm)
    return 4/(np.sqrt(k) * frobenius_mtm) * ( np.trace(mtm) * M @ M.T @ M / frobenius_mtm**2 - M)

def opt_M_for_data(X, m_init, lr=0.001, iters=1, regularization=0.01, opt_loops=100):
    '''
    Given data X, find the orthogonal matrix M which minimizes the sum norm of X's singular tubes. The algorithm
    runs a projected gradient descent type algorithm which has many opt_loops. In each opt_loop, M is orthogonalized,
    then a sequence of gradient descent steps improve M's compression of X and perturbs its orthogonality.

    Args:
        X: an n1 x n2 x n3 tensor
        m_init: an n3 x n3 orthogonal matrix, to be used for initializing gradient descent
        lr: the learning rate parameter for gradient descent
        iters: number of descent steps to take per loop. Each descent step updates M better compress X, though it
            may become non-orthogonal.
        regularization: strength of orthogonality penalization, encourages M to be almost orthogonal after each iter
        opt_loops: number of sequences of gradient descent iterations.

    Returns:
        A n3 x n3 orthogonal matrix which well compressed X
    '''

    def orthogonalize(M, mode="qr"):
        if(mode == "svd"):
            U, S, V = np.linalg.svd(M)
            return U @ V.T
        elif(mode == "qr"):
            Q, R = np.linalg.qr(M)
            return Q

    lmbd = lr
    gamma = regularization
    m, p, n = X.shape
    M = np.copy(m_init)
    k = M.shape[0]
    coord = (1, 1, 1)
    coord_over_time = []
    sumnorm_over_time = []
    M_pre_ortho = None
    for i in range(opt_loops):
        Xhat = x3(X, M)
        Uh, S, V = m_SVD(X, M)

        def outer(A, B):
            return np.tensordot(A[..., None], B, (-1, -1))

        for j in trange(iters):
            Uh_hat, Shat, Vhat = x3(Uh, M), x3(S, M), x3(V, M)
            Spr_hat = xFace(xFace(Uh_hat, Xhat), Vhat)

            print(f"Sum norm over time: {np.sum(np.abs(Spr_hat))}")
            sumnorm_over_time.append(np.sum(np.abs(Spr_hat)))

            def fast_grad_trnsf(i, j):
                grad_from_prods = Uh[:, :, i] @ Xhat[:, :, j] @ Vhat[:, :, j] + \
                                  Uh_hat[:, :, j] @ X[:, :, i] @ Vhat[:, :, j] + \
                                  Uh_hat[:, :, j] @ Xhat[:, :, j] @ V[:, :, i]
                return np.sum(np.sign(Spr_hat[:, :, j]) * grad_from_prods) / (m*p*n)
            # note: dtype is for dtype of the indices, not the output
            grad = np.fromfunction(np.frompyfunc(fast_grad_trnsf, 2, 1), shape=(k, k), dtype=np.int)
            # output is dtype 'object' and should be float64

            M = M - lmbd * grad.astype(np.complex64)
            coord_over_time.append(Spr_hat[coord])
        M_pre_ortho = np.copy(M)
        M = orthogonalize(M)

    Uh, S, V = m_SVD(X, M)
    Uhat, Shat, Vhat = x3(Uh, M), x3(S, M), x3(V, M)
    r, c = 3, 2


    def imshow_cpx_safe(M):
        if(not np.all(np.isreal(M))):
            plt.imshow(np.abs(M))
        else:
            plt.imshow(M.astype(np.float32))

    plt.subplot(r, c, 1)
    plt.title("Orthogonalized M")
    imshow_cpx_safe(M)
    plt.colorbar()

    plt.subplot(r, c, 2)

    plt.title("Pre-orthogonalized M")
    imshow_cpx_safe(M_pre_ortho)
    plt.colorbar()

    plt.subplot(r, c, 3)
    plt.title("Singular Tubes as Rows")
    rows = np.vstack([S[None, i, i, :] for i in range(min(m, p))])
    imshow_cpx_safe(rows)
    plt.colorbar()

    plt.subplot(r, c, 4)
    plt.title("Singular Tubes as Rows (Transform Domain)")
    rows_hat = np.vstack([Shat[None, i, i, :] for i in range(min(m, p))])
    imshow_cpx_safe(rows_hat)
    plt.colorbar()

    plt.subplot(r, c, 5)
    plt.title("S[1, 1, 1] Over Time")
    plt.plot(coord_over_time)
    plt.title(f"S{str(list(coord))} over Optimization")
    plt.xlabel("Optimization Step")
    plt.ylabel("Value")

    plt.subplot(r, c, 6)
    plt.title("Sum Norm over Optimization")
    plt.plot(sumnorm_over_time)
    plt.xlabel("Optimization Step")
    plt.ylabel("Value")

    plt.suptitle(f"Optimizing M: $\\lambda={lmbd}$, $\\gamma={gamma}$, itr={iters}, opt_loops={opt_loops}", y =0.99)
    plt.gcf().set_size_inches(15, 15)
    plt.tight_layout()
    plt.savefig("Optimization.png", dpi=120)
    plt.show()
    return M



class TestTensorGradients(unittest.TestCase):
    def setUp(self):
        self.shapes = ( 2, 3, 5, 4, 6 )
        n1, n2, n3, n4, n5 = self.shapes
        A = np.random.uniform(size=(n1, n2, n4))
        B = np.random.uniform(size=(n2, n3, n4))
        C = np.arange(n3 * n5 * n4).reshape((n3, n5, n4)) / (n3 * n4 * n5)
        self.A = A
        self.B = B
        self.C = C
        self.eps = 0.000001

    def test_smg_identity(self):
        n1, n2, n3, n4, n5 = self.shapes
        A, B, C = self.A, self.B, self.C
        i, j = 0, 1
        M = np.eye(n4)
        Mpr = np.copy(M)
        Mpr[i, j] += self.eps
        numeric_grad = (xM(A, B, Mpr) - xM(A, B, M)) / self.eps
        analytic_grad = star_m_grad(A, B, M, (i, j))
        self.assertTrue(np.allclose(numeric_grad, analytic_grad, rtol=10e-4))

    def test_smg_haar(self):
        n1, n2, n3, n4, n5 = self.shapes
        A, B, C = self.A, self.B, self.C
        i, j = 0, 1
        M = haarmtx(n4, normalize=True)
        Mpr = np.copy(M)
        Mpr[i, j] += self.eps
        numeric_grad = (xM(A, B, Mpr) - xM(A, B, M)) / self.eps
        analytic_grad = star_m_grad(A, B, M, (i, j))
        self.assertTrue(np.allclose(numeric_grad, analytic_grad, rtol=10e-4))

    def test_l1smg_identity(self):
        n1, n2, n3, n4, n5 = self.shapes
        A, B, C = self.A, self.B, self.C
        i, j = 0, 1
        M = np.eye(n4)
        Mpr = np.copy(M)
        Mpr[i, j] += self.eps
        f_xpr = xM(xM(A, B, Mpr), C, Mpr)
        f_x = xM(xM(A, B, M), C, M)
        numeric_grad = (np.sum(np.abs(f_xpr)) - np.sum(np.abs(f_x))) / self.eps
        analytic_grad = sum_norm_m_grad(A, B, C, M, (i, j))
        self.assertTrue(np.allclose(numeric_grad, analytic_grad, rtol=10e-4))

    def test_l1smg_haar(self):
        n1, n2, n3, n4, n5 = self.shapes
        A, B, C = self.A, self.B, self.C
        i, j = 0, 1
        M = haarmtx(n4)
        Mpr = np.copy(M)
        Mpr[i, j] += self.eps
        f_xpr = xM(xM(A, B, Mpr), C, Mpr)
        f_x = xM(xM(A, B, M), C, M)
        numeric_grad = (np.sum(np.abs(f_xpr)) - np.sum(np.abs(f_x))) / self.eps
        analytic_grad = sum_norm_m_grad(A, B, C, M, (i, j))
        self.assertTrue(np.allclose(numeric_grad, analytic_grad, rtol=10e-4))

    def test_tsmg_identity(self):
        n1, n2, n3, n4, n5 = self.shapes
        A, B, C = self.A, self.B, self.C
        i, j = 0, 1
        M = np.eye(n4)
        Mpr = np.copy(M)
        Mpr[i, j] += self.eps
        numeric_grad = (xM(xM(A, B, Mpr), C, Mpr) - xM(xM(A, B, M), C, M)) / self.eps
        analytic_grad = three_star_m_grad(A, B, C, M, (i, j))
        self.assertTrue(np.allclose(numeric_grad, analytic_grad, rtol=10e-4))

    def test_tsmg_haar(self):
        n1, n2, n3, n4, n5 = self.shapes
        A, B, C = self.A, self.B, self.C
        i, j = 0, 1
        M = haarmtx(n4, normalize=True)
        Mpr = np.copy(M)
        Mpr[i, j] += self.eps
        numeric_grad = (xM(xM(A, B, Mpr), C, Mpr) - xM(xM(A, B, M), C, M)) / self.eps
        analytic_grad = three_star_m_grad(A, B, C, M, (i, j))
        self.assertTrue(np.allclose(numeric_grad, analytic_grad, rtol=10e-4))

    def test_nonortho_gradient_identity(self):
        n1 = self.shapes[0]
        i, j = 0, 1
        M = np.eye(n1)
        Mpr = np.copy(M)
        Mpr[i, j] += self.eps

        def penalty(M):
            return np.sum( (M.T @ M / np.linalg.norm(M.T @ M) - np.eye(n1) / np.sqrt(n1)) ** 2)
        numeric_grad = (penalty(Mpr) - penalty(M)) / self.eps
        analytic_grad = nonortho_penalty_grad(M)[i, j]
        self.assertTrue(np.allclose(numeric_grad, analytic_grad, rtol=10e-4, atol=1e-6))

    def test_nonortho_gradient_haar(self):
        n1 = self.shapes[0]
        i, j = 0, 1
        M = haarmtx(n1, normalize=True)
        Mpr = np.copy(M)
        Mpr[i, j] += self.eps

        def penalty(M):
            return np.sum( (M.T @ M / np.linalg.norm(M.T @ M) - np.eye(n1) / np.sqrt(n1)) ** 2)
        numeric_grad = (penalty(Mpr) - penalty(M)) / self.eps
        analytic_grad = nonortho_penalty_grad(M)
        self.assertTrue(np.allclose(numeric_grad, analytic_grad[i,j], rtol=10e-4, atol=1e-6))

    def test_nonortho_gradient_random(self):
        n1 = self.shapes[0]
        i, j = 0, 1
        M = np.random.normal(size=(n1, n1))
        Mpr = np.copy(M)
        Mpr[i, j] += self.eps

        def penalty(M):
            return np.sum( (M.T @ M / np.linalg.norm(M.T @ M) - np.eye(n1) / np.sqrt(n1)) ** 2)
        numeric_grad = (penalty(Mpr) - penalty(M)) / self.eps
        analytic_grad = nonortho_penalty_grad(M)
        self.assertTrue(np.allclose(numeric_grad, analytic_grad[i,j], rtol=10e-4, atol=1e-6))

if __name__ == "__main__":
    cats_R = np.load("data/Cats_RGB.npy")[:, :, :, 0]
    # compressible data via random rank 1 outer products *weighted* with decaying weights
    opt_M_for_data(cats_R, haarmtx(32, normalize=True), lr=0.1, iters=250, opt_loops=1, regularization=0)
    unittest.main()

