import numpy as np
import matplotlib.pyplot as plt
from tensors import m_SVD, haarmtx, dctmtx, dftmtx, xM, x3, safe_inv

def truncate(A, M, k):
    Minv = safe_inv(M)
    Uh, Sh, Vh = m_SVD(A, M, inv_transform=False)
    flat_coeff_idxs = np.argsort(Sh, axis=None)[::-1]
    coeff_idxs = np.unravel_index(flat_coeff_idxs, Sh.shape)
    coeffs = tuple(r[:k] for r in coeff_idxs)
    mask = np.zeros_like(Sh)
    mask[coeffs] = 1
    Sh_truncated = mask * Sh
    U, S_truncated, V = x3(Uh, Minv), x3(Sh_truncated, Minv), x3(Vh, Minv)
    Vt = V.transpose((1, 0, 2))
    return xM(U, xM(S_truncated, Vt, M), M)

def compression_ratio(k, m, p, n):
    return (m*p*n) / (k*(m + p + 1))

def k_for_compression_ratio(c, m, p, n):
    return (m*p*n)/(c*(m+p+1))

def RSE(A, At):
    norm = np.sqrt(np.sum(A**2))
    err = np.sqrt(np.sum((At - A)**2))
    return 20 * np.log10(err / norm)

if __name__ == "__main__":
    A = np.load("Escalator.npy")
    A = A[:, :, :128] # truncate n so haar mtx is applicable
    A = (A - np.min(A))/np.ptp(A)
    m, p, n = A.shape
    nnz = n*min(m, p)

    M = dctmtx(n)
    H = haarmtx(n)
    F = dftmtx(n)

    compress = [int(np.ceil(k_for_compression_ratio(x, m, p, n))) for x in range(2, 11)]
    ratios = [compression_ratio(r, m, p, n) for r in compress]
    dct_rse = [RSE(A, truncate(A, M, r)) for r in compress]
    haar_rse = [RSE(A, truncate(A, H, r)) for r in compress]
    dft_rse = [RSE(A, truncate(A, F, r)) for r in compress]

    plt.plot(ratios, dct_rse, "ro-", label=r'$\star_M$ DCT')
    plt.plot(ratios, haar_rse, "b^-", label=r'$\star_H$ Haar')
    plt.plot(ratios, dft_rse, "gs-", label=r'$\star_F$ Fourier')
    plt.xlabel("Compression Ratio")
    plt.ylabel("RSE (dB, lower is better)")
    plt.title("Haar vs. DCT Compression Performance - Elevator Video")
    plt.legend()
    plt.savefig("Haar_vs_DCT_Elevator.png")
    plt.show()

