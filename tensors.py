import numpy as np
import unittest
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import cm
import json
from scipy.fftpack import dct, fft, ifft
import pywt
import traceback

CACHED_M = None

class set_M():
    def __init__(self, M):
        self.M = M
    def __enter__(self):
        global CACHED_M
        assert_mat(self.M)
        assert np.linalg.cond(self.M) < np.inf, "M must be invertible"
        CACHED_M = self.M
    def __exit__(self, exc_type, exc_value, tb):
        global CACHED_M
        CACHED_M = None
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)


def assert_vec(x):
    assert len(x.shape) == 2
    assert x.shape[1] == 1  # row vectors

def assert_mat(x):
    assert len(x.shape) == 2

def _fetch(M):
    global CACHED_M
    if M is None:
        if CACHED_M is None:
            raise ValueError("Cannot get M as it is currently unspecified.")
        else:
            return CACHED_M
    else:
        assert_mat(M)
        assert np.linalg.cond(M) < np.inf, "M must be invertible"
        return M

def assert_tensor(x, order=3):
    assert (len(x.shape) == order), f"x must be an order-{order} tensor"


def assert_real(M):
    assert not np.iscomplex(M).any()


def normF(A):
    return np.sqrt(np.sum(A ** 2))

def normN(A, M = None):
    M = _fetch(M)
    Uh, Sh, Vh = m_SVD(A, M, inv_transform=False)
    return np.sum(Sh)

def as_vec(A):
    return np.flatten(A)


def as_mat(A, m, n):
    return np.reshape((A), (m, n), "F")


def twist(A):
    assert_mat(A)
    return A[:, None, :]


def sq(A):
    assert_tensor(A, order=3)
    assert A.shape[1] == 1
    return A[:, 0, :]


def safe_inv(M):
    assert M.shape[0] == M.shape[1], "M must be a square matrix"
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        raise AssertionError("M must be invertible")
    return Minv


def apply_inv(A, M = None):
    M = _fetch(M)
    return np.linalg.solve(M, A)


def cT(A, M = None):
    M = _fetch(M)
    '''
    Conjugate transpose
    '''
    assert_mat(M)
    assert_tensor(A, 3)
    Ah = x3(A, M)
    Ah_hermitian = Ah.conj().transpose((1, 0, 2))
    return np.real_if_close(x3_inv(Ah_hermitian, M), tol=1000)


def rT(A, M = None):
    '''
    Real transpose
    '''
    return np.copy(A).transpose((1, 0, 2))


def mode_unf(A, mode):
    assert_tensor(A, order=3)
    assert mode <= 3 and mode >= 1

    if (mode == 1):
        return np.concatenate(np.transpose(A, (2, 0, 1)), axis=1)
    elif (mode == 2):
        return np.concatenate(np.transpose(A, (2, 0, 1)), axis=0).T
    elif (mode == 3):
        return np.concatenate(np.transpose(A, (1, 0, 2)), axis=0).T


def mode_fold(A, shape, mode):
    assert_mat(A)
    assert mode <= 3 and mode >= 1

    m, p, n = shape
    if (mode == 1):
        return np.stack([A[:, i * p:(i + 1) * p] for i in range(n)], axis=2)
    elif (mode == 2):
        return np.stack([A[:, i * m:(i + 1) * m].T for i in range(n)], axis=2)
    elif (mode == 3):
        return np.stack([A[:, i * m:(i + 1) * m].T for i in range(p)], axis=1)


def mode_prod(A, M=None, mode=1):
    M = _fetch(M)
    assert_tensor(A, order=3)
    assert_mat(M)
    assert M.shape[1] == A.shape[mode - 1], f"Incorrect dimensions for {A.shape} x{mode} {M.shape}"

    s = list(A.shape)
    s[mode - 1] = M.shape[0]

    return mode_fold(np.matmul(M, mode_unf(A, mode)), s, mode)


def mode_inv(A, M=None, mode=1):
    M = _fetch(M)
    assert_tensor(A, order=3)
    assert_mat(M)
    assert M.shape[1] == A.shape[mode - 1], f"Incorrect dimensions for {A.shape} x{mode} {M.shape}"
    assert M.shape[0] == M.shape[1], f"Only square matrices are invertible"
    return mode_fold(apply_inv(mode_unf(A, mode), M), A.shape, mode)


def x3(A, M=None):
    M = _fetch(M)
    return mode_prod(A, M, mode=3)


def x2(A, M=None):
    M = _fetch(M)
    return mode_prod(A, M, mode=2)


def x1(A, M=None):
    M = _fetch(M)
    return mode_prod(A, M, mode=1)


def x3_inv(A, M=None):
    M = _fetch(M)
    return mode_inv(A, M, mode=3)


def x2_inv(A, M=None):
    M = _fetch(M)
    return mode_inv(A, M, mode=2)


def x1_inv(A, M=None):
    M = _fetch(M)
    return mode_inv(A, M, mode=1)


def xFace(A, B):
    assert_tensor(A, order=3)
    assert_tensor(B, order=3)
    m, n, p = A.shape
    q, r, s = B.shape
    assert n == q, "Modes 1 and 2 of tensors A and B must have compatible dimensions for matrix multiplication"
    assert p == s, "Mode-3 of tensors A and B must be equal"

    C = Parallel(n_jobs=4, prefer="threads")(delayed(np.matmul)(a, b) for a, b \
                                             in zip(A.transpose((2, 0, 1)), B.transpose((2, 0, 1))))
    return np.stack(C, axis=-1)


def xM(A, B, M=None):
    M = _fetch(M)
    assert_tensor(A, order=3)
    assert_tensor(B, order=3)
    assert_mat(M)

    m, n, p = A.shape
    q, r, s = B.shape
    assert n == q, "Modes 1 and 2 of tensors A and B must have compatible dimensions for matrix multiplication"
    assert p == s, "Mode-3 of tensors A and B must be equal"
    assert p == M.shape[0], "Dimensions of M must match mode-3 dimension of A, B"

    Ah = x3(A, M)
    Bh = x3(B, M)

    Ch = xFace(Ah, Bh)
    return x3_inv(Ch, M)


def matrix_algebra(M):
    assert_mat(M)

    Minv = safe_inv(M)
    mt = M.T[:, None, :]
    mt_inv = Minv[None, :, :]
    return xFace(mt, mt_inv).transpose((0, 2, 1))


def explicit_matrix_algebra(M):
    assert_mat(M)
    Minv = safe_inv(M)
    mt = M.T
    mit = Minv.T
    faces = [np.kron(mt[:, i].reshape((-1, 1)), mit[i, :].reshape((1, -1))) for i in range(len(M))]
    return np.stack(faces, axis=-1).transpose((0, 2, 1))


def fft_H(M):
    Mt = M.T
    Mt[:, :, 1:] = Mt[:, :, 1:][:, :, ::-1]  # reverse all put the first items of mode 3
    return Mt


def m_SVD(A, M=None, inv_transform=True):
    M = _fetch(M)
    assert_tensor(A, order=3)
    assert_mat(M)

    Ah = x3(A, M)
    Ah_svd = Parallel(n_jobs=4, prefer="threads")(delayed(np.linalg.svd)(a) for a in Ah.transpose((2, 0, 1)))
    u, s, vt = zip(*Ah_svd)
    Uh = np.stack(u, axis=-1)
    Sh = np.zeros_like(Ah)
    for i in range(len(s)):
        np.fill_diagonal(Sh[:, :, i], s[i])
    Vh = np.stack(vt, axis=-1).transpose((1, 0, 2))
    if (inv_transform):
        U, S, V = x3_inv(Uh, M), x3_inv(Sh, M), x3_inv(Vh, M)
        return np.real_if_close(U, tol=1000), np.real_if_close(S, tol=1000), np.real_if_close(V, tol=1000)
    else:
        return Uh, Sh, Vh


def idftmtx(n, normalize=False):
    d = ifft(np.eye(n), axis=0)
    if (normalize):
        d = d / np.sqrt(np.sum(d ** 2, axis=1)).reshape((-1, 1))
    return d


def dftmtx(n, normalize=False):
    d = fft(np.eye(n), axis=0)
    if (normalize):
        d = d / np.sqrt(np.sum(np.abs(d) ** 2, axis=1)).reshape((-1, 1))
    return d


def dctmtx(n, normalize=False):
    d = dct(np.eye(n), axis=0)
    if (normalize):
        d = d / np.sqrt(np.sum(d ** 2, axis=1)).reshape((-1, 1))
    return d


def dyadic_decomp(r):
    r = np.array(r).flatten()
    assert np.log2(len(r)) == np.ceil(np.log2(len(r))), "Input must have length 2**k"

    def _dyad(r, a=[]):
        if (len(r) == 1):
            return [r] + a  # concatenation
        return _dyad(r[:len(r) // 2], [r[len(r) // 2:]] + a)

    return _dyad(r)


def wavelet_mtx(s, wvt):
    assert np.log2(s) == np.ceil(np.log2(s))

    def _dyad(r, a=[]):
        if (len(r) == 1):
            return [r] + a  # concatenation
        return _dyad(r[:len(r) // 2], [r[len(r) // 2:]] + a)

    coeffs = [_dyad(r) for r in np.eye(s).astype(np.int64)]
    return np.vstack([pywt.waverec(r, wvt) for r in coeffs])


def haarmtx(n, normalize=False):
    assert np.log2(n) == np.floor(np.log2(n)), "Haar matrices must be of side length 2**n"
    k = np.floor(np.log2(n))

    def _haarmtx(k, mtx):
        if (k == 0):
            return mtx
        else:
            _mtx = np.concatenate(
                (np.kron(mtx, [1, 1]),  # dilate by factor of 2
                 np.kron(np.eye(len(mtx)), [1, -1])),
                # dilate + alternate the identity, ie. translates of highest frequency
                axis=0)
            return _haarmtx(k - 1, _mtx)

    h = _haarmtx(k, [[1, ]])
    if (normalize):
        h = h / np.sqrt(np.sum(h ** 2, axis=1)).reshape((-1, 1))
    return h


def to_voxels(A):
    assert_tensor(A, order=3)
    m, p, n = A.shape
    data = {"dimension": [{"height": m, "width": p, "depth": n}],
            "voxels": []}

    intensities = (np.copy(A) - np.min(A)) / np.ptp(A)
    cmap = cm.get_cmap('viridis')
    for i in range(m):
        print(i)
        for j in range(p):
            for k in range(n):
                v_id = f"voxel_{i * n * p + j * n + k}"
                x, y, z = j, i, k
                c = cmap(intensities[i, j, k])
                data["voxels"].append({
                    "id": v_id,
                    "x": x,
                    "y": y,
                    "z": z,
                    "red": int(255 * c[0]),
                    "green": int(255 * c[1]),
                    "blue": int(255 * c[2])
                })
    return data


def tensor_to_json(A, path):
    if (len(A.shape) == 4):
        assert A.shape[3] == 3, "A must have 3 channels in each tensor cell"
        m, p, n, _ = A.shape
        colors = (A - np.min(A)) / np.ptp(A)
    else:
        m, p, n = A.shape
        colors = np.zeros((m, p, n, 3))
        if (np.iscomplex(A).any()):
            intensities = np.angle(A)
            intensities[intensities < 0] += 2 * np.pi
            intensities /= 2 * np.pi
            cmap = cm.get_cmap('twilight')  # cyclic
        else:
            intensities = (A - np.min(A)) / np.ptp(A)
            cmap = cm.get_cmap('viridis')

        for i in range(m):
            for j in range(p):
                for k in range(n):
                    x, y, z = j, i, k
                    c = cmap(intensities[i, j, k])
                    colors[i, j, k] = np.array((c[0], c[1], c[2]))

    colors_arr = [[[[round(float(c), 3) for c in colors[i, j, k]] for k in range(n)] for j in range(p)] for i in
                  range(m)]
    with open(path, "w+") as f_out:
        f_out.write(json.dumps(colors_arr))


class TestTensorProducts(unittest.TestCase):
    def setUp(self):
        A = np.zeros((2, 3, 4))
        for i in range(A.shape[2]):
            A[:, :, i] = i + 1
        self.A = A

    def test_fold_m1(self):
        A = self.A
        self.assertTrue((mode_fold(mode_unf(A, 1), A.shape, 1) == A).all())

    def test_fold_m2(self):
        A = self.A
        self.assertTrue((mode_fold(mode_unf(A, 2), A.shape, 2) == A).all())

    def test_fold_m3(self):
        A = self.A
        self.assertTrue((mode_fold(mode_unf(A, 3), A.shape, 3) == A).all())

    def test_prod_m1(self):
        A = self.A
        m, p, n = A.shape
        self.assertTrue((x1(A, np.eye(m)) == A).all())

    def test_prod_inv(self):
        B_face = np.array([[1, 4], [3, 1]])
        B_face_inv = np.linalg.inv(B_face)
        I_face = np.eye(2)

        B = np.stack((B_face,) * 3, axis=2)
        I = np.stack((I_face,) * 3, axis=2)

        c = x1(B, B_face_inv)
        self.assertTrue(np.allclose(x1(B, B_face_inv), I))
        self.assertTrue(np.allclose(x2(B, B_face_inv.T), I))

    def test_prod_m2(self):
        A = self.A
        m, p, n = A.shape
        self.assertTrue((x2(A, np.eye(p)) == A).all())

    def test_prod_m1(self):
        A = self.A
        m, p, n = A.shape
        self.assertTrue((x3(A, np.eye(n)) == A).all())

    def test_fourier_m(self):
        X = np.random.normal(size=(2, 3, 8))
        M = dftmtx(8, normalize=True)
        U, S, V = m_SVD(X, M)
        self.assertTrue(np.allclose(X, xM(xM(U, S, M), cT(V, M), M)))


if __name__ == "__main__":
    unittest.main()
