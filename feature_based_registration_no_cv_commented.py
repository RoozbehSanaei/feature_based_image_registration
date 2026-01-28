#!/usr/bin/env python3
"""
Feature-based 2D image registration (no computer vision libraries).

What this file does
-------------------
Given two grayscale images (source and target), it estimates a geometric transform that maps
points from the source image into the target image using a classic feature pipeline:

1) Detect keypoints (corners)               : Harris / Shi-Tomasi / FAST-9
2) Estimate an orientation for each keypoint: quick gradient-based orientation
3) Describe each keypoint                    : SIFT-like (128D), raw normalized patch, or BRIEF (binary)
4) Match descriptors                         : ratio test + optional mutual (cross-check)
5) Robust model fitting                      : RANSAC (similarity / affine / homography)

Constraints and intent
----------------------
- This implementation uses only NumPy (and Matplotlib only for demo plots).
- The goal is educational and "reasonably robust", not speed-optimized production code.
- There is no subpixel localization, no scale-space extrema refinement, and no sophisticated NMS.
- The SIFT-like descriptor is simplified for clarity and uses hard cell assignment.

Typical usage
-------------
Call `register_pair(img1, img2, cfg)` with a `RegistrationConfig`.
It returns the estimated 3x3 transform H (homogeneous coordinates) and inlier diagnostics.

Notes
-----
- All images are expected to be float32 arrays in [0, 1] (the demo uses synthetic images).
- Homographies require at least 4 matches, affine 3, similarity 2.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Literal

import numpy as np

# -----------------------------------------------------------------------------
# Data structures and configuration
# -----------------------------------------------------------------------------

@dataclass
class Keypoint:
    """
    A 2D keypoint with a scale and an orientation.

    x, y
        Pixel coordinates in image space.

    sigma
        Scale parameter used for smoothing and/or descriptor sampling.

    response
        Detector response used for ranking and thresholding.

    angle
        Orientation in radians used for rotation-invariant descriptors (optional).
    """
    x: float
    y: float
    sigma: float
    response: float
    angle: float = 0.0


DetectorType = Literal["harris", "shi_tomasi", "fast9"]
DescriptorType = Literal["sift128", "patch", "brief"]
ModelType = Literal["similarity", "affine", "homography"]
MatchMetric = Literal["l2", "hamming"]


@dataclass
class DetectorConfig:
    """
    Keypoint detector configuration.

    For Harris/Shi-Tomasi:
    - sigmas: list of smoothing scales (roughly a tiny multi-scale detector)
    - rel_threshold: threshold as a fraction of max response at each scale
    - max_per_scale: max corners kept per sigma after NMS/ranking

    For FAST9:
    - fast_threshold: intensity threshold in [0, 1] for circle comparisons
    - fast_border: pixels to avoid near image boundary
    """
    kind: DetectorType = "harris"
    sigmas: Tuple[float, ...] = (1.4, 2.2)
    max_kps_total: int = 260
    max_per_scale: int = 140
    nms_radius: int = 3
    rel_threshold: float = 0.015
    fast_threshold: float = 0.10
    fast_border: int = 4


@dataclass
class SiftDescConfig:
    """
    Simplified SIFT-like descriptor settings.

    num_cells x num_cells spatial grid (default 4x4)
    num_bins orientation histogram bins per cell (default 8)
    cell_size size in pixels of each cell (default 4), so window size is 16x16
    step_scale controls sampling step vs sigma
    """
    num_cells: int = 4
    num_bins: int = 8
    cell_size: int = 4
    step_scale: float = 1.0


@dataclass
class PatchDescConfig:
    """Normalized patch descriptor settings."""
    patch_radius: int = 11
    blur_sigma: float = 1.0


@dataclass
class BriefConfig:
    """
    BRIEF descriptor settings.

    nbits: number of binary tests (256 gives 32 bytes).
    patch_radius: sampling region radius around keypoint.
    rotate: if True, rotate BRIEF test pairs using keypoint orientation.
    """
    nbits: int = 256
    patch_radius: int = 15
    seed: int = 13
    rotate: bool = True
    blur_sigma: float = 1.0


@dataclass
class DescriptorConfig:
    """Descriptor selection and per-descriptor settings."""
    kind: DescriptorType = "sift128"
    sift: SiftDescConfig = field(default_factory=SiftDescConfig)
    patch: PatchDescConfig = field(default_factory=PatchDescConfig)
    brief: BriefConfig = field(default_factory=BriefConfig)


@dataclass
class MatchConfig:
    """
    Matching settings.

    ratio: Lowe's ratio threshold (smaller -> stricter).
    mutual: if True, apply cross-check (mutual nearest neighbors).
    top_k: keep at most this many best matches by distance.
    """
    ratio: float = 0.75
    mutual: bool = True
    top_k: int = 2500


@dataclass
class RansacConfig:
    """
    RANSAC settings for robust model estimation.

    iters: number of random hypotheses.
    threshold_px: inlier threshold in pixels (for geometric error).
    symmetric: if True, use symmetric error (forward + backward).
    sampler: "prosac" favors early matches (sorted by descriptor distance).
    """
    iters: int = 700
    threshold_px: float = 3.0
    seed: int = 0
    symmetric: bool = True
    sampler: Literal["uniform", "prosac"] = "prosac"
    prosac_step: int = 25
    prosac_min_pool: int = 40


@dataclass
class RegistrationConfig:
    """
    Top-level registration configuration.

    quick_orientation:
        Fast orientation estimate at keypoint using local image gradients.
        This enables rotation-invariant descriptors and BRIEF rotation.
    """
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    descriptor: DescriptorConfig = field(default_factory=DescriptorConfig)
    matcher: MatchConfig = field(default_factory=MatchConfig)
    model: ModelType = "homography"
    ransac: RansacConfig = field(default_factory=RansacConfig)
    quick_orientation: bool = True


# -----------------------------------------------------------------------------
# Low-level image operations (Gaussian blur + Sobel gradients)
# -----------------------------------------------------------------------------

def gaussian_kernel1d(sigma: float) -> np.ndarray:
    """
    Create a 1D Gaussian kernel with radius ~3*sigma.

    Using separability (blur x then y) keeps the implementation simple and avoids
    2D kernel generation.
    """
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    r = int(math.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return (k / (k.sum() + 1e-12)).astype(np.float32)


def convolve1d_reflect(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """
    1D convolution with 'reflect' padding.

    Parameters
    ----------
    img : (H, W) float32
    k   : 1D kernel
    axis: 0 for y-direction, 1 for x-direction

    Notes
    -----
    - This is intentionally straightforward (loop over rows/cols).
    - For real-time speed, you'd use FFT or vectorized convolution.
    """
    r = (len(k) - 1) // 2
    pad = [(0, 0)] * 2
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="reflect")

    out = np.empty_like(img, dtype=np.float32)
    if axis == 0:
        for i in range(img.shape[0]):
            out[i, :] = (p[i:i + 2 * r + 1, :] * k[:, None]).sum(axis=0)
    else:
        for j in range(img.shape[1]):
            out[:, j] = (p[:, j:j + 2 * r + 1] * k[None, :]).sum(axis=1)
    return out


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur."""
    k = gaussian_kernel1d(sigma)
    return convolve1d_reflect(convolve1d_reflect(img, k, 1), k, 0)


def sobel_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate image gradients using Sobel filters.

    Returns
    -------
    gx, gy : gradients in x and y directions.
    """
    kx = np.array([1, 0, -1], dtype=np.float32)
    ky = np.array([1, 2, 1], dtype=np.float32)

    # gx: blur in y then derivative in x
    tmp = convolve1d_reflect(img, ky, 0)
    gx = convolve1d_reflect(tmp, kx, 1)

    # gy: blur in x then derivative in y
    tmp = convolve1d_reflect(img, ky, 1)
    gy = convolve1d_reflect(tmp, kx, 0)
    return gx, gy


# -----------------------------------------------------------------------------
# Corner detectors: Harris / Shi-Tomasi / FAST-9
# -----------------------------------------------------------------------------

def structure_tensor(img: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute smoothed second-moment matrix entries.

    M = [[Ix^2, IxIy],
         [IxIy, Iy^2]]

    We compute Ix, Iy on a blurred image, then smooth Ix^2, Iy^2, IxIy again.
    """
    b = gaussian_blur(img, sigma)
    gx, gy = sobel_gradients(b)
    ixx, iyy, ixy = gx * gx, gy * gy, gx * gy
    s = max(1.0, 1.5 * sigma)
    return gaussian_blur(ixx, s), gaussian_blur(iyy, s), gaussian_blur(ixy, s)


def harris_response(img: np.ndarray, sigma: float, k: float = 0.04) -> np.ndarray:
    """Harris corner response: det(M) - k * trace(M)^2."""
    ixx, iyy, ixy = structure_tensor(img, sigma)
    det = ixx * iyy - ixy * ixy
    tr = ixx + iyy
    return (det - k * (tr * tr)).astype(np.float32)


def shitomasi_response(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Shi-Tomasi corner measure: min eigenvalue of M.

    For 2x2 matrix M, eigenvalues are:
      λ = 0.5 * (trace ± sqrt(trace^2 - 4*det))
    We take the smaller eigenvalue.
    """
    ixx, iyy, ixy = structure_tensor(img, sigma)
    tr = ixx + iyy
    det = ixx * iyy - ixy * ixy
    disc = np.maximum(tr * tr - 4 * det, 0.0)
    return (0.5 * (tr - np.sqrt(disc))).astype(np.float32)


def nonmax_suppression(R: np.ndarray, radius: int, thr: float) -> List[Tuple[int, int, float]]:
    """
    Simple non-maximum suppression (NMS) in a square window.

    Returns a list of (y, x, value) peaks.

    Notes
    -----
    - This is a naive implementation (nested loops).
    - For faster NMS, you'd use dilation / max filters.
    """
    H, W = R.shape
    r = radius
    peaks: List[Tuple[int, int, float]] = []
    for y in range(r, H - r):
        for x in range(r, W - r):
            v = float(R[y, x])
            if v < thr:
                continue
            patch = R[y - r:y + r + 1, x - r:x + r + 1]
            if v >= float(patch.max()):
                peaks.append((y, x, v))
    return peaks


# FAST-9 circle offsets (radius 3 pixels around center).
_FAST_CIRCLE = np.array([
    (0, -3), (1, -3), (2, -2), (3, -1),
    (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1),
    (-3, 0), (-3, -1), (-2, -2), (-1, -3)
], dtype=np.int32)


def detect_fast9(img: np.ndarray, threshold: float, border: int) -> List[Tuple[int, int, float]]:
    """
    FAST-9 corner detector.

    A pixel is a corner if there exists a contiguous arc of length 9 on the circle
    where all pixels are either > p + t OR < p - t.

    Returns
    -------
    corners: list of (y, x, score). Score here is a simple sum of abs differences.
    """
    H, W = img.shape
    t = float(threshold)
    corners: List[Tuple[int, int, float]] = []

    # Quick rejection using 4 points on the circle (0, 4, 8, 12).
    quick = [0, 4, 8, 12]

    for y in range(border, H - border):
        for x in range(border, W - border):
            p = float(img[y, x])

            # Quick test: if fewer than 3 of the 4 are brighter/darker, reject.
            vals = [float(img[y + dy, x + dx]) for (dy, dx) in _FAST_CIRCLE[quick]]
            if sum(v > p + t for v in vals) < 3 and sum(v < p - t for v in vals) < 3:
                continue

            # Full circle test.
            circ = np.array([img[y + dy, x + dx] for (dy, dx) in _FAST_CIRCLE], dtype=np.float32)
            hi = circ > p + t
            lo = circ < p - t

            # Wrap-around check by concatenating a prefix.
            hi2 = np.r_[hi, hi[:8]]
            lo2 = np.r_[lo, lo[:8]]

            ok = False
            for s in range(16):
                if hi2[s:s + 9].all() or lo2[s:s + 9].all():
                    ok = True
                    break
            if not ok:
                continue

            # A simple corner "strength" heuristic.
            score = float(np.abs(circ - p).sum())
            corners.append((y, x, score))

    return corners


def detect_keypoints(img: np.ndarray, cfg: DetectorConfig) -> List[Keypoint]:
    """
    Detect and rank keypoints according to the selected detector.

    Output is globally capped at cfg.max_kps_total.
    """
    kps: List[Keypoint] = []

    if cfg.kind in ("harris", "shi_tomasi"):
        # Multi-scale: compute response at each sigma, run NMS, take top per scale.
        for s in cfg.sigmas:
            R = harris_response(img, s) if cfg.kind == "harris" else shitomasi_response(img, s)
            R = np.maximum(R, 0.0)  # keep positive responses only
            thr = cfg.rel_threshold * float(R.max() + 1e-12)

            peaks = nonmax_suppression(R, cfg.nms_radius, thr)
            peaks.sort(key=lambda t: t[2], reverse=True)
            peaks = peaks[:cfg.max_per_scale]

            for y, x, v in peaks:
                kps.append(Keypoint(float(x), float(y), float(s), float(v)))

    else:
        # FAST-9: returns many candidates; apply a simple "suppression" mask.
        corners = detect_fast9(img, cfg.fast_threshold, cfg.fast_border)
        corners.sort(key=lambda t: t[2], reverse=True)

        taken = np.zeros(img.shape, dtype=np.uint8)
        r = cfg.nms_radius

        for y, x, sc in corners:
            if taken[y, x]:
                continue

            # Mark a neighborhood as taken (very simple spatial suppression).
            taken[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 1

            kps.append(Keypoint(float(x), float(y), 1.6, float(sc)))
            if len(kps) >= cfg.max_kps_total:
                break

    # Global cap, keep strongest keypoints.
    kps.sort(key=lambda k: k.response, reverse=True)
    return kps[:cfg.max_kps_total]


# -----------------------------------------------------------------------------
# Keypoint orientation (quick gradient-based)
# -----------------------------------------------------------------------------

def quick_orientations(img: np.ndarray, kps: List[Keypoint], blur_sigma: float = 1.2) -> List[Keypoint]:
    """
    Estimate keypoint orientations using local gradients.

    For each keypoint, we blur the image, compute Sobel gradients, and take a small
    3x3 mean around the keypoint location. Orientation is atan2(gy, gx).

    This is much cheaper than a full SIFT orientation histogram, but it is also less robust.
    """
    b = gaussian_blur(img, blur_sigma)
    gx, gy = sobel_gradients(b)

    out: List[Keypoint] = []
    for kp in kps:
        x = int(round(kp.x))
        y = int(round(kp.y))
        if x < 1 or y < 1 or x >= img.shape[1] - 1 or y >= img.shape[0] - 1:
            continue

        # Local averaging stabilizes orientation slightly.
        gxx = float(gx[y - 1:y + 2, x - 1:x + 2].mean())
        gyy = float(gy[y - 1:y + 2, x - 1:x + 2].mean())
        ang = math.atan2(gyy, gxx)

        out.append(Keypoint(kp.x, kp.y, kp.sigma, kp.response, ang))

    return out


# -----------------------------------------------------------------------------
# Descriptors: SIFT-like (128), normalized patch, BRIEF (256)
# -----------------------------------------------------------------------------

def _brief_pairs(nbits: int, patch_radius: int, seed: int) -> np.ndarray:
    """Generate BRIEF test pairs (dx1, dy1, dx2, dy2) within [-r, r]."""
    rng = np.random.default_rng(seed)
    r = patch_radius
    return rng.integers(-r, r + 1, size=(nbits, 4), dtype=np.int32)


# Precomputed popcount table for 8-bit integers (for fast Hamming distance).
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _pairwise_hamming(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between packed binary descriptors."""
    x = np.bitwise_xor(P1[:, None, :], P2[None, :, :])
    return _POPCOUNT[x].sum(axis=2).astype(np.int32)


def _pairwise_l2(D1: np.ndarray, D2: np.ndarray) -> np.ndarray:
    """
    Pairwise squared Euclidean distances using:
      ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    """
    n1 = (D1 * D1).sum(axis=1, keepdims=True)
    n2 = (D2 * D2).sum(axis=1, keepdims=True).T
    return np.maximum(n1 + n2 - 2 * (D1 @ D2.T), 0.0)


def sift128_from_grad(mag: np.ndarray, ang: np.ndarray, kp: Keypoint, cfg: SiftDescConfig) -> Optional[np.ndarray]:
    """
    Build a simplified SIFT-like descriptor (see header docstring for details).

    Returns 128D descriptor or None if the sampling window would go out of bounds.
    """
    win = cfg.num_cells * cfg.cell_size
    half = win / 2.0
    step = max(1.0, cfg.step_scale * kp.sigma)

    coords = (np.arange(win, dtype=np.float32) + 0.5 - half)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")

    ca, sa = math.cos(kp.angle), math.sin(kp.angle)
    dx = (ca * xx - sa * yy) * step
    dy = (sa * xx + ca * yy) * step

    xs = kp.x + dx
    ys = kp.y + dy

    H, W = mag.shape
    if xs.min() < 1 or ys.min() < 1 or xs.max() >= W - 2 or ys.max() >= H - 2:
        return None

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    ax = (xs - x0).astype(np.float32)
    ay = (ys - y0).astype(np.float32)

    def bilinear(M: np.ndarray) -> np.ndarray:
        v00 = M[y0, x0]
        v10 = M[y0, x0 + 1]
        v01 = M[y0 + 1, x0]
        v11 = M[y0 + 1, x0 + 1]
        return (1 - ax) * (1 - ay) * v00 + ax * (1 - ay) * v10 + (1 - ax) * ay * v01 + ax * ay * v11

    smag = bilinear(mag).astype(np.float32)
    sang = bilinear(ang).astype(np.float32)
    rel = (sang - kp.angle + 2 * np.pi) % (2 * np.pi)

    # Gaussian weighting in the descriptor window
    w_sigma = 0.5 * win * step
    w = np.exp(-((xx * step) ** 2 + (yy * step) ** 2) / (2 * w_sigma * w_sigma)).astype(np.float32)
    smag *= w

    cell = cfg.cell_size
    ci = (np.arange(win) // cell).astype(np.int32)
    cj = (np.arange(win) // cell).astype(np.int32)
    CII, CJJ = np.meshgrid(ci, cj, indexing="ij")

    bins = cfg.num_bins
    bin_f = (rel / (2 * np.pi)) * bins
    b0 = (np.floor(bin_f).astype(np.int32)) % bins
    frac = (bin_f - np.floor(bin_f)).astype(np.float32)
    b1 = (b0 + 1) % bins

    desc = np.zeros((cfg.num_cells, cfg.num_cells, bins), dtype=np.float32)
    for i in range(cfg.num_cells):
        for j in range(cfg.num_cells):
            m = (CII == i) & (CJJ == j)
            if not np.any(m):
                continue
            mm = smag[m]
            bb0 = b0[m]
            bb1 = b1[m]
            ff = frac[m]
            np.add.at(desc[i, j], bb0, (1 - ff) * mm)
            np.add.at(desc[i, j], bb1, ff * mm)

    v = desc.reshape(-1)
    v = v / (np.linalg.norm(v) + 1e-12)
    v = np.clip(v, 0.0, 0.2)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32)


def patch_descriptor(img: np.ndarray, kp: Keypoint, cfg: PatchDescConfig, blur_img: np.ndarray) -> Optional[np.ndarray]:
    """Extract a normalized patch descriptor around the keypoint."""
    r = cfg.patch_radius
    x0 = int(round(kp.x))
    y0 = int(round(kp.y))
    if x0 - r < 1 or y0 - r < 1 or x0 + r >= img.shape[1] - 1 or y0 + r >= img.shape[0] - 1:
        return None

    p = blur_img[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1].astype(np.float32)
    p -= float(p.mean())
    p /= (float(np.linalg.norm(p)) + 1e-12)
    return p.reshape(-1)


def brief_descriptor(img: np.ndarray, kp: Keypoint, cfg: BriefConfig, pairs: np.ndarray, blur_img: np.ndarray) -> Optional[np.ndarray]:
    """Compute a BRIEF descriptor (packed bits) at the keypoint."""
    r = cfg.patch_radius
    x0 = float(kp.x)
    y0 = float(kp.y)
    if x0 - r - 1 < 0 or y0 - r - 1 < 0 or x0 + r + 2 >= img.shape[1] or y0 + r + 2 >= img.shape[0]:
        return None

    if cfg.rotate:
        ca, sa = math.cos(kp.angle), math.sin(kp.angle)
        p = pairs.astype(np.float32)
        x1, y1, x2, y2 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        off = np.stack([
            ca * x1 - sa * y1, sa * x1 + ca * y1,
            ca * x2 - sa * y2, sa * x2 + ca * y2
        ], axis=1)
    else:
        off = pairs.astype(np.float32)

    xs1 = x0 + off[:, 0]
    ys1 = y0 + off[:, 1]
    xs2 = x0 + off[:, 2]
    ys2 = y0 + off[:, 3]

    def bilinear(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        x0i = np.floor(xs).astype(np.int32)
        y0i = np.floor(ys).astype(np.int32)
        ax = (xs - x0i).astype(np.float32)
        ay = (ys - y0i).astype(np.float32)

        x0i = np.clip(x0i, 0, blur_img.shape[1] - 2)
        y0i = np.clip(y0i, 0, blur_img.shape[0] - 2)

        v00 = blur_img[y0i, x0i]
        v10 = blur_img[y0i, x0i + 1]
        v01 = blur_img[y0i + 1, x0i]
        v11 = blur_img[y0i + 1, x0i + 1]
        return (1 - ax) * (1 - ay) * v00 + ax * (1 - ay) * v10 + (1 - ax) * ay * v01 + ax * ay * v11

    v1 = bilinear(xs1, ys1)
    v2 = bilinear(xs2, ys2)
    bits = (v1 < v2)

    packed = np.packbits(bits.astype(np.uint8), bitorder="little")
    return packed.astype(np.uint8)


def compute_descriptors(img: np.ndarray, kps: List[Keypoint], cfg: DescriptorConfig) -> Tuple[List[Keypoint], np.ndarray, MatchMetric]:
    """Compute descriptors for a list of keypoints, returning only valid ones."""
    kept: List[Keypoint] = []
    descs: List[np.ndarray] = []

    if cfg.kind == "sift128":
        sigs = sorted(set([float(k.sigma) for k in kps]))
        grad_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        for s in sigs:
            b = gaussian_blur(img, s)
            gx, gy = sobel_gradients(b)
            mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
            ang = np.arctan2(gy, gx).astype(np.float32)
            grad_cache[s] = (mag, ang)

        for kp in kps:
            mag, ang = grad_cache[float(kp.sigma)]
            d = sift128_from_grad(mag, ang, kp, cfg.sift)
            if d is None:
                continue
            kept.append(kp)
            descs.append(d)

        if not descs:
            dim = cfg.sift.num_cells * cfg.sift.num_cells * cfg.sift.num_bins
            return [], np.zeros((0, dim), dtype=np.float32), "l2"
        return kept, np.stack(descs, axis=0), "l2"

    if cfg.kind == "patch":
        blur_img = gaussian_blur(img, cfg.patch.blur_sigma) if cfg.patch.blur_sigma > 0 else img
        for kp in kps:
            d = patch_descriptor(img, kp, cfg.patch, blur_img)
            if d is None:
                continue
            kept.append(kp)
            descs.append(d)

        if not descs:
            dim = (2 * cfg.patch.patch_radius + 1) ** 2
            return [], np.zeros((0, dim), dtype=np.float32), "l2"
        return kept, np.stack(descs, axis=0), "l2"

    if cfg.kind == "brief":
        blur_img = gaussian_blur(img, cfg.brief.blur_sigma) if cfg.brief.blur_sigma > 0 else img
        pairs = _brief_pairs(cfg.brief.nbits, cfg.brief.patch_radius, cfg.brief.seed)

        for kp in kps:
            d = brief_descriptor(img, kp, cfg.brief, pairs, blur_img)
            if d is None:
                continue
            kept.append(kp)
            descs.append(d)

        if not descs:
            return [], np.zeros((0, cfg.brief.nbits // 8), dtype=np.uint8), "hamming"
        return kept, np.stack(descs, axis=0), "hamming"

    raise ValueError(f"Unknown descriptor kind: {cfg.kind}")


# -----------------------------------------------------------------------------
# Matching (ratio + mutual)
# -----------------------------------------------------------------------------

def match_ratio_mutual(
    D1: np.ndarray,
    D2: np.ndarray,
    metric: MatchMetric,
    ratio: float,
    mutual: bool,
    top_k: int
) -> List[Tuple[int, int, float]]:
    """Compute tentative matches using ratio test and optional mutual NN check."""
    if D1.shape[0] == 0 or D2.shape[0] == 0:
        return []

    if metric == "l2":
        dist = np.sqrt(_pairwise_l2(D1.astype(np.float32), D2.astype(np.float32)) + 1e-12)
    else:
        dist = _pairwise_hamming(D1.astype(np.uint8), D2.astype(np.uint8)).astype(np.float32)

    idx = np.argsort(dist, axis=1)
    if idx.shape[1] < 2:
        return []

    j1, j2 = idx[:, 0], idx[:, 1]
    d1 = dist[np.arange(dist.shape[0]), j1]
    d2 = dist[np.arange(dist.shape[0]), j2] + 1e-12

    keep = d1 <= float(ratio) * d2
    matches = [(i, int(j1[i]), float(d1[i])) for i in range(D1.shape[0]) if keep[i]]

    if mutual and matches:
        best_i_for_j = dist.argmin(axis=0)
        matches = [(i, j, d) for (i, j, d) in matches if best_i_for_j[j] == i]

    matches.sort(key=lambda t: t[2])
    if top_k and len(matches) > top_k:
        matches = matches[:top_k]
    return matches


# -----------------------------------------------------------------------------
# Geometry helpers + model fitting
# -----------------------------------------------------------------------------

def to_h(pts: np.ndarray) -> np.ndarray:
    """Convert Nx2 points to homogeneous Nx3 points."""
    return np.c_[pts, np.ones((pts.shape[0], 1), dtype=np.float64)]


def proj(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 3x3 transform to Nx2 points and dehomogenize."""
    q = (H @ to_h(pts).T).T
    return q[:, :2] / (q[:, 2:3] + 1e-12)


def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalization (translate + scale) to improve numerical stability."""
    mean = pts.mean(axis=0)
    cen = pts - mean
    rms = np.sqrt((cen ** 2).sum(axis=1).mean()) + 1e-12
    s = math.sqrt(2.0) / rms
    T = np.array([[s, 0, -s * mean[0]],
                  [0, s, -s * mean[1]],
                  [0, 0, 1]], dtype=np.float64)
    pn = (T @ to_h(pts).T).T
    return T, pn


def fit_homography(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """DLT homography fit with point normalization."""
    T1, p1 = normalize_points(pts1.astype(np.float64))
    T2, p2 = normalize_points(pts2.astype(np.float64))
    N = pts1.shape[0]

    A = np.zeros((2 * N, 9), dtype=np.float64)
    for i in range(N):
        x, y, w = p1[i]
        u, v, t = p2[i]
        A[2 * i] = [0, 0, 0, -x, -y, -w, v * x, v * y, v * w]
        A[2 * i + 1] = [x, y, w, 0, 0, 0, -u * x, -u * y, -u * w]

    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1
    return H / (H[2, 2] + 1e-12)


def fit_affine(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Least-squares affine fit in normalized coordinates, then denormalize."""
    T1, p1 = normalize_points(pts1.astype(np.float64))
    T2, p2 = normalize_points(pts2.astype(np.float64))
    x, y = p1[:, 0], p1[:, 1]
    u, v = p2[:, 0], p2[:, 1]
    N = pts1.shape[0]

    A = np.zeros((2 * N, 6), dtype=np.float64)
    b = np.zeros((2 * N,), dtype=np.float64)

    A[0::2, 0] = x
    A[0::2, 1] = y
    A[0::2, 2] = 1
    A[1::2, 3] = x
    A[1::2, 4] = y
    A[1::2, 5] = 1

    b[0::2] = u
    b[1::2] = v

    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b1, tx, c, d, ty = p

    An = np.array([[a, b1, tx],
                   [c, d, ty],
                   [0, 0, 1]], dtype=np.float64)

    H = np.linalg.inv(T2) @ An @ T1
    return H / (H[2, 2] + 1e-12)


def fit_similarity(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Similarity fit via Umeyama alignment."""
    X = pts1.astype(np.float64)
    Y = pts2.astype(np.float64)

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY

    varX = (Xc * Xc).sum() / (X.shape[0] + 1e-12)
    Sigma = (Yc.T @ Xc) / (X.shape[0] + 1e-12)

    U, S, Vt = np.linalg.svd(Sigma)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    s = S.sum() / (varX + 1e-12)
    t = muY - s * (R @ muX)

    H = np.array([[s * R[0, 0], s * R[0, 1], t[0]],
                  [s * R[1, 0], s * R[1, 1], t[1]],
                  [0, 0, 1]], dtype=np.float64)
    return H


def fit_model(model: ModelType, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Model dispatcher."""
    if model == "homography":
        return fit_homography(pts1, pts2)
    if model == "affine":
        return fit_affine(pts1, pts2)
    if model == "similarity":
        return fit_similarity(pts1, pts2)
    raise ValueError(model)


def min_samples(model: ModelType) -> int:
    """Minimal number of point pairs needed per model."""
    return {"similarity": 2, "affine": 3, "homography": 4}[model]


# -----------------------------------------------------------------------------
# Robust fitting (RANSAC) + error metrics
# -----------------------------------------------------------------------------

def sym_error(H: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Symmetric transfer error: ||p2-H(p1)|| + ||p1-H^{-1}(p2)||."""
    e_f = np.linalg.norm(p2 - proj(H, p1), axis=1)
    Hinv = np.linalg.inv(H)
    e_b = np.linalg.norm(p1 - proj(Hinv, p2), axis=1)
    return e_f + e_b


def fwd_error(H: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Forward transfer error: ||p2-H(p1)||."""
    return np.linalg.norm(p2 - proj(H, p1), axis=1)


def ransac_fit(
    p1: np.ndarray,
    p2: np.ndarray,
    model: ModelType,
    cfg: RansacConfig,
    prior_order: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC robust estimation with optional PROSAC-like sampling.

    Returns a refined model fit to inliers of the best hypothesis.
    """
    N = p1.shape[0]
    m = min_samples(model)
    rnd = random.Random(cfg.seed)
    prior_order = np.arange(N) if prior_order is None else prior_order

    best_H = None
    best_in = None
    best_cnt = -1
    best_cost = float("inf")

    for it in range(int(cfg.iters)):
        if cfg.sampler == "prosac":
            pool = min(N, int(cfg.prosac_min_pool + it // max(1, cfg.prosac_step)))
            pool = max(pool, m)
            cand = prior_order[:pool].tolist()
            sample = rnd.sample(cand, m)
        else:
            sample = rnd.sample(range(N), m)

        try:
            H = fit_model(model, p1[sample], p2[sample])
        except Exception:
            continue

        errs = sym_error(H, p1, p2) if cfg.symmetric else fwd_error(H, p1, p2)
        inl = errs < float(cfg.threshold_px)
        cnt = int(inl.sum())

        # MSAC-like cost prefers many inliers and smaller inlier errors.
        cost = float(np.minimum(errs * errs, cfg.threshold_px ** 2).sum())

        if (cnt > best_cnt) or (cnt == best_cnt and cost < best_cost):
            best_H = H
            best_in = inl
            best_cnt = cnt
            best_cost = cost

    if best_H is None or best_in is None:
        raise RuntimeError("RANSAC failed")

    H_ref = fit_model(model, p1[best_in], p2[best_in])
    return H_ref, best_in


# -----------------------------------------------------------------------------
# High-level pipeline
# -----------------------------------------------------------------------------

def register_pair(img1: np.ndarray, img2: np.ndarray, cfg: RegistrationConfig) -> Dict[str, object]:
    """
    Full feature-based registration pipeline.

    Returns a dict with the estimated transform and diagnostic info.
    """
    I1 = img1.astype(np.float32)
    I2 = img2.astype(np.float32)

    # 1) Detect
    k1 = detect_keypoints(I1, cfg.detector)
    k2 = detect_keypoints(I2, cfg.detector)

    # 2) Orient
    if cfg.quick_orientation:
        k1 = quick_orientations(I1, k1, blur_sigma=1.2)
        k2 = quick_orientations(I2, k2, blur_sigma=1.2)

    # 3) Describe
    k1, D1, metric = compute_descriptors(I1, k1, cfg.descriptor)
    k2, D2, _ = compute_descriptors(I2, k2, cfg.descriptor)

    # 4) Match
    matches = match_ratio_mutual(D1, D2, metric, cfg.matcher.ratio, cfg.matcher.mutual, cfg.matcher.top_k)
    if len(matches) < max(8, 2 * min_samples(cfg.model)):
        raise RuntimeError("Too few matches")

    p1 = np.array([[k1[i].x, k1[i].y] for i, j, d in matches], dtype=np.float64)
    p2 = np.array([[k2[j].x, k2[j].y] for i, j, d in matches], dtype=np.float64)

    dists = np.array([d for _, _, d in matches], dtype=np.float64)
    prior = np.argsort(dists)

    # 5) Robust fit
    H, inl = ransac_fit(p1, p2, cfg.model, cfg.ransac, prior_order=prior)

    errs = sym_error(H, p1, p2) if cfg.ransac.symmetric else fwd_error(H, p1, p2)
    return {"H": H, "inliers": inl, "matches": matches, "kps1": k1, "kps2": k2, "errs": errs}
