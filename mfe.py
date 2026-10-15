"""
M-polar fuzzy image enhancement using exposure fusion.

Python port of the MATLAB class `expFusion`:
  1. Build m fuzzy images (the "poles") from the input with the tan-gamma
     generator   F_k = 1 - (1 - A) / (1 + tan(gamma_k) * A).
  2. Compute per-pole quality maps (contrast, saturation, well-exposedness),
     combine them into a weight map and normalise across the poles.
  3. Blend the poles with Laplacian-pyramid exposure fusion (Mertens et al.).

Run as a script:
    python mfe.py [image] [out.png]
"""
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

DEFAULT_GAMMAS = (-0.75, -0.6, -0.3, 0.0, 1.2, 1.4, 1.5)   # radians, as in the MATLAB code
DEFAULT_M = (1.0, 1.0, 1.0)      # exponents: (contrast, saturation, well-exposedness)
DEFAULT_EXPOSED = 0.5
SIGMA = 0.2
DEFAULT_IMAGE = Path(__file__).with_name("sample.png")
_PYR = np.array([.0625, .25, .375, .25, .0625])
_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def to_float(A):
    A = np.asarray(A)
    if A.dtype == np.uint8:
        return A.astype(float) / 255.0
    if A.dtype == np.uint16:
        return A.astype(float) / 65535.0
    return np.clip(A.astype(float), 0, 1)


def entropy(img):
    h, _ = np.histogram(np.clip(img, 0, 1), bins=256, range=(0, 1))
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def rgb2gray(I):
    return I[..., 0] * 0.2989 + I[..., 1] * 0.5870 + I[..., 2] * 0.1140


# ----------------------------------------------------------------------------
# m-polar fuzzy images
# ----------------------------------------------------------------------------
def fuzzy_image(Mem, gamma):
    denom = np.maximum(1.0 + np.tan(gamma) * Mem, np.finfo(float).eps)
    return np.clip(1.0 - (1.0 - Mem) / denom, 0.0, 1.0)


def m_polar_images(A, gammas=DEFAULT_GAMMAS):
    """Return array of shape (m, H, W, 3): one fuzzy image per pole."""
    Mem = to_float(A)
    return np.stack([fuzzy_image(Mem, g) for g in gammas])


# ----------------------------------------------------------------------------
# quality measures
# ----------------------------------------------------------------------------
def contrast(I):
    return np.stack([np.abs(ndi.correlate(rgb2gray(x), _LAPLACIAN, mode="nearest")) for x in I])


def saturation(I):
    out = []
    for x in I:
        mu = x.mean(axis=2, keepdims=True)
        out.append(np.sqrt(((x - mu) ** 2).sum(axis=2) / 3))
    return np.stack(out)


def well_exposedness(I, exp_value=DEFAULT_EXPOSED):
    return np.exp(-0.5 * (I - exp_value) ** 2 / SIGMA ** 2).prod(axis=3)


# ----------------------------------------------------------------------------
# pyramids (direct port of the MATLAB routines)
# ----------------------------------------------------------------------------
def _filt(I, mode):
    for ax in (0, 1):
        I = ndi.correlate1d(I, _PYR, axis=ax, mode=mode)
    return I


def _down(I):
    return _filt(I, "reflect")[::2, ::2]


def _up(I, odd):
    I = np.pad(I, [(1, 1), (1, 1)] + [(0, 0)] * (I.ndim - 2), mode="edge")
    r, c = 2 * I.shape[0], 2 * I.shape[1]
    R = np.zeros((r, c) + I.shape[2:])
    R[::2, ::2] = 4 * I
    R = _filt(R, "constant")
    return R[2:r - 2 - odd[0], 2:c - 2 - odd[1]]


def _nlev(shape):
    return int(np.floor(np.log2(min(shape[:2]))))


def gaussian_pyramid(I):
    pyr = [I]
    for _ in range(_nlev(I.shape)):
        I = _down(I)
        pyr.append(I)
    return pyr


def laplacian_pyramid(I):
    pyr, J = [], I
    for _ in range(_nlev(I.shape)):
        K = _down(J)
        odd = (2 * K.shape[0] - J.shape[0], 2 * K.shape[1] - J.shape[1])
        pyr.append(J - _up(K, odd))
        J = K
    pyr.append(J)
    return pyr


def reconstruct(pyr):
    R = pyr[-1]
    for l in range(len(pyr) - 2, -1, -1):
        odd = (2 * R.shape[0] - pyr[l].shape[0], 2 * R.shape[1] - pyr[l].shape[1])
        R = pyr[l] + _up(R, odd)
    return R


# ----------------------------------------------------------------------------
# fusion
# ----------------------------------------------------------------------------
def weight_maps(I, m=DEFAULT_M, exp_value=DEFAULT_EXPOSED):
    W = np.ones(I.shape[:3])
    if m[0] > 0:
        W = W * contrast(I) ** m[0]
    if m[1] > 0:
        W = W * saturation(I) ** m[1]
    if m[2] > 0:
        W = W * well_exposedness(I, exp_value) ** m[2]
    W = W + 1e-12
    return W / W.sum(axis=0, keepdims=True)


def exposure_fusion(I, m=DEFAULT_M, exp_value=DEFAULT_EXPOSED, return_weights=False):
    W = weight_maps(I, m, exp_value)
    pyr = None
    for i in range(I.shape[0]):
        pW = gaussian_pyramid(W[i])
        pI = laplacian_pyramid(I[i])
        if pyr is None:
            pyr = [np.zeros_like(p) for p in pI]
        for l in range(len(pyr)):
            pyr[l] += pW[l][..., None] * pI[l]
    out = np.clip(reconstruct(pyr), 0, 1)
    return (out, W) if return_weights else out


def fuzzy_exposure_fusion(A, m=DEFAULT_M, gammas=DEFAULT_GAMMAS, exp_value=DEFAULT_EXPOSED,
                          return_details=False):
    """Full pipeline. A: RGB image (uint8 or float in [0,1]). Returns float RGB in [0,1]."""
    I = m_polar_images(A, gammas)
    out, W = exposure_fusion(I, m, exp_value, return_weights=True)
    return (out, I, W) if return_details else out


def main():
    from skimage import io
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    dst = sys.argv[2] if len(sys.argv) > 2 else "enhanced_mpolar.png"
    A = io.imread(src)[..., :3]
    out = fuzzy_exposure_fusion(A)
    io.imsave(dst, np.rint(out * 255).astype(np.uint8))
    print(f"poles (m) = {len(DEFAULT_GAMMAS)}, entropy {entropy(rgb2gray(to_float(A))):.4f} -> "
          f"{entropy(rgb2gray(out)):.4f}; saved {dst}")


if __name__ == "__main__":
    main()
