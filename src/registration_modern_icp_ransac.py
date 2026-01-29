#!/usr/bin/env python3
"""
3D point registration (rigid SE(3)) with modern Python structure.

What this module provides
-------------------------
(2) ICP family
    - point-to-point ICP (baseline)
    - trimmed ICP (drop worst correspondences each iteration)
    - point-to-plane ICP (optional; needs target normals)

(3) Robust global registration
    - RANSAC over *provided* correspondences (e.g., feature matches)

Design principles used in this file
-----------------------------------
- Clear separation of concerns (math, NN, solvers, synthetic tests).
- Strong typing (TypeAlias + numpy.typing) to document array intent.
- Dataclasses with slots for small immutable parameter/result carriers.
- Deterministic experiments via explicit RNG seeds.
- Minimal dependencies (numpy + scipy) and no hidden global state.

Practical note
--------------
Nearest-neighbor correspondences without a coarse pose or features are fragile.
This file includes NN-based correspondence building only as a baseline.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Types: make intent explicit and catch shape/precision mistakes early in review
# ---------------------------------------------------------------------------

F64: TypeAlias = np.float64
I64: TypeAlias = np.int64

Points: TypeAlias = NDArray[F64]  # Expected shape: (N, 3)
Mat4: TypeAlias = NDArray[F64]    # Expected shape: (4, 4)
Mat3: TypeAlias = NDArray[F64]    # Expected shape: (3, 3)
Vec3: TypeAlias = NDArray[F64]    # Expected shape: (3,)


# ---------------------------------------------------------------------------
# Small helper: normalize dtype at boundaries
# Principle: Do conversions at module boundaries, not inside tight loops.
# ---------------------------------------------------------------------------

def _as_f64(x: NDArray) -> NDArray[F64]:
    """Convert input array-like to contiguous float64 numpy array."""
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# SE(3) transform utilities
# Principle: keep transforms in one canonical representation (4x4 homogeneous).
# ---------------------------------------------------------------------------

def make_transform(R: Mat3, t: Vec3) -> Mat4:
    """
    Build a 4x4 homogeneous transform from rotation R and translation t.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _as_f64(R)
    T[:3, 3] = _as_f64(t).reshape(3)
    return T


def apply_transform(T: Mat4, pts: Points) -> Points:
    """
    Apply a homogeneous transform to a set of 3D points.

    Implementation detail:
    - Uses homogeneous coordinates for clarity and to avoid manual broadcasting.
    """
    pts = _as_f64(pts)
    ph = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    return (T @ ph.T).T[:, :3]


def compose(T_new: Mat4, T_old: Mat4) -> Mat4:
    """
    Compose transforms: return T = T_new @ T_old.

    Meaning:
    - If p' = T_old p and p'' = T_new p', then p'' = (T_new @ T_old) p.
    """
    return _as_f64(T_new) @ _as_f64(T_old)


# ---------------------------------------------------------------------------
# Rigid alignment with known correspondences (Kabsch / SVD)
# Principle: keep the foundational solver small, pure, and well-tested.
# ---------------------------------------------------------------------------

def kabsch_se3(A: Points, B: Points) -> Mat4:
    """
    Least-squares rigid transform A->B given correspondences A[i] <-> B[i].

    This is the core primitive used by ICP and by RANSAC refinement.
    """
    A = _as_f64(A)
    B = _as_f64(B)

    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError("A and B must have shape (N,3) and match.")

    # Center both sets to decouple translation from rotation.
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    AA = A - cA
    BB = B - cB

    # SVD of covariance gives optimal rotation in least-squares sense.
    U, _, Vt = np.linalg.svd(AA.T @ BB)
    R = Vt.T @ U.T

    # Reflection fix: enforce det(R)=+1 for a proper rotation.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cB - R @ cA
    return make_transform(R, t)


def pose_error(T_est: Mat4, T_gt: Mat4) -> tuple[float, float]:
    """
    Compare estimated and ground-truth transforms.

    Returns:
      (rotation_angle_radians, translation_error_norm)

    Rotation angle is computed from the trace of the relative rotation matrix.
    """
    Terr = np.linalg.inv(_as_f64(T_est)) @ _as_f64(T_gt)
    R = Terr[:3, :3]
    t = Terr[:3, 3]
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    return angle, float(np.linalg.norm(t))


# ---------------------------------------------------------------------------
# Rotation parametrization helpers (for point-to-plane linearization + synth init)
# Principle: isolate numerically delicate math in small, reviewable functions.
# ---------------------------------------------------------------------------

def skew(v: Vec3) -> Mat3:
    """Skew-symmetric matrix [v]_x such that [v]_x p = v × p."""
    x, y, z = map(float, _as_f64(v).reshape(3))
    return np.array([[0.0, -z,  y],
                     [z,  0.0, -x],
                     [-y, x,  0.0]], dtype=np.float64)


def rodrigues(w: Vec3) -> Mat3:
    """
    Exponential map from so(3) vector w to SO(3) rotation matrix.

    Used for:
    - Turning small rotation updates into a rotation matrix in point-to-plane ICP
    - Generating controlled perturbations for synthetic tests
    """
    w = _as_f64(w).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    k = w / theta
    K = skew(k)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


# ---------------------------------------------------------------------------
# Nearest-neighbor search wrapper
# Principle: build KD-tree once when possible; avoid rebuilding per-iteration.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NNResult:
    """Nearest-neighbor query result for a batch of points."""
    dists: NDArray[F64]     # Shape: (N,)
    indices: NDArray[I64]   # Shape: (N,), indices into target


def nn_query(tree: cKDTree, src: Points) -> NNResult:
    """
    Query a prebuilt KD-tree for the closest target point to each source point.
    """
    d, idx = tree.query(_as_f64(src), k=1)
    return NNResult(
        dists=_as_f64(d).reshape(-1),
        indices=np.asarray(idx, dtype=np.int64).reshape(-1),
    )


# ---------------------------------------------------------------------------
# ICP (Iterative Closest Point)
# Principle: expose parameters in a dataclass, return results in a dataclass.
# ---------------------------------------------------------------------------

class ICPMode(str, Enum):
    """ICP objective variant."""
    POINT_TO_POINT = "p2p"
    POINT_TO_PLANE = "p2l"


@dataclass(frozen=True, slots=True)
class ICPParams:
    """
    ICP configuration.

    trim_ratio:
      - 0.0 means keep all matches
      - e.g. 0.3 means drop 30% worst matches each iteration (robustness)
    """
    max_iter: int = 60
    tol: float = 1e-7
    trim_ratio: float = 0.0
    mode: ICPMode = ICPMode.POINT_TO_POINT


@dataclass(frozen=True, slots=True)
class ICPResult:
    """ICP output bundle."""
    T: Mat4
    rmse: float
    iters: int
    converged: bool


def _trim_indices(dists: NDArray[F64], trim_ratio: float) -> NDArray[I64]:
    """
    Select indices to keep after trimming worst matches.

    Principle:
    - centralize input validation and selection logic so the solver stays readable.
    """
    if trim_ratio <= 0.0:
        return np.arange(dists.size, dtype=np.int64)
    if not (0.0 <= trim_ratio < 0.95):
        raise ValueError("trim_ratio must be in [0, 0.95).")

    keep = int(np.ceil((1.0 - trim_ratio) * dists.size))
    return np.argsort(dists)[:keep].astype(np.int64)


def icp_point_to_point(
    source: Points,
    target: Points,
    *,
    init_T: Mat4 | None = None,
    params: ICPParams = ICPParams(mode=ICPMode.POINT_TO_POINT),
) -> ICPResult:
    """
    Point-to-point ICP:
    - alternates between nearest-neighbor matching and Kabsch rigid update.

    Convergence criterion:
    - absolute change in RMSE falls below params.tol
    """
    src = _as_f64(source)
    dst = _as_f64(target)

    # KD-tree built once (critical for performance).
    tree = cKDTree(dst)

    # Start from identity if no initial guess provided.
    T = np.eye(4, dtype=np.float64) if init_T is None else _as_f64(init_T)
    prev = np.inf

    for it in range(1, params.max_iter + 1):
        src_t = apply_transform(T, src)

        nn = nn_query(tree, src_t)
        sel = _trim_indices(nn.dists, params.trim_ratio)

        # Build correspondence sets and solve best-fit rigid transform.
        A = src_t[sel]
        B = dst[nn.indices[sel]]
        T_delta = kabsch_se3(A, B)

        # Update transform (apply delta after current estimate).
        T = compose(T_delta, T)

        # Track error on the selected set (trimmed if enabled).
        d = nn.dists[sel]
        rmse = float(np.sqrt(np.mean(d * d)))

        if abs(prev - rmse) < params.tol:
            return ICPResult(T=T, rmse=rmse, iters=it, converged=True)
        prev = rmse

    return ICPResult(T=T, rmse=float(prev), iters=params.max_iter, converged=False)


def icp_point_to_plane(
    source: Points,
    target: Points,
    target_normals: Points,
    *,
    init_T: Mat4 | None = None,
    params: ICPParams = ICPParams(mode=ICPMode.POINT_TO_PLANE),
) -> ICPResult:
    """
    Point-to-plane ICP:
    - minimizes n_i^T (R p_i + t - q_i) using a small-angle linearization.

    Practical use:
    - faster convergence than point-to-point when normals are accurate.
    - requires target normals aligned with target points.
    """
    src = _as_f64(source)
    dst = _as_f64(target)
    nrm = _as_f64(target_normals)

    if dst.shape != nrm.shape:
        raise ValueError("target and target_normals must have same shape (M,3).")

    tree = cKDTree(dst)

    T = np.eye(4, dtype=np.float64) if init_T is None else _as_f64(init_T)
    prev = np.inf

    for it in range(1, params.max_iter + 1):
        src_t = apply_transform(T, src)
        nn = nn_query(tree, src_t)
        sel = _trim_indices(nn.dists, params.trim_ratio)

        # Matched points and normals
        p = src_t[sel]
        q = dst[nn.indices[sel]]
        n = nrm[nn.indices[sel]]

        # Residual: r_i = n_i^T(p_i - q_i)
        r = np.einsum("ij,ij->i", n, (p - q))

        # Linearized system A x = -r where x = [w, dt] (6 DOF)
        # The rotation part uses identity: n^T (w × p) = (p × n)^T w
        pxn = np.cross(p, n)
        A = np.hstack([pxn, n])
        b = (-r).reshape(-1, 1)

        # Least squares solve for small update
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        w = x[:3].reshape(3)
        dt = x[3:].reshape(3)

        # Convert small rotation vector to rotation matrix
        dR = rodrigues(w)

        T_delta = make_transform(dR, dt)
        T = compose(T_delta, T)

        d = nn.dists[sel]
        rmse = float(np.sqrt(np.mean(d * d)))

        if abs(prev - rmse) < params.tol:
            return ICPResult(T=T, rmse=rmse, iters=it, converged=True)
        prev = rmse

    return ICPResult(T=T, rmse=float(prev), iters=params.max_iter, converged=False)


def icp(
    source: Points,
    target: Points,
    *,
    init_T: Mat4 | None = None,
    params: ICPParams = ICPParams(),
    target_normals: Points | None = None,
) -> ICPResult:
    """
    Dispatch wrapper for ICP modes.

    Principle:
    - callers select mode via params.mode; this function enforces required inputs.
    """
    match params.mode:
        case ICPMode.POINT_TO_POINT:
            return icp_point_to_point(source, target, init_T=init_T, params=params)
        case ICPMode.POINT_TO_PLANE:
            if target_normals is None:
                raise ValueError("target_normals is required for point-to-plane ICP.")
            return icp_point_to_plane(source, target, target_normals, init_T=init_T, params=params)
        case _:
            raise ValueError(f"Unknown ICP mode: {params.mode}")


# ---------------------------------------------------------------------------
# RANSAC (robust global registration from correspondences)
# Principle: RANSAC should be fed *putative correspondences* from features/matching.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RANSACParams:
    """RANSAC configuration."""
    threshold: float = 0.05
    max_iter: int = 3000
    confidence: float = 0.999
    seed: int = 0


@dataclass(frozen=True, slots=True)
class RANSACResult:
    """RANSAC output bundle."""
    T: Mat4
    inliers: NDArray[np.bool_]
    num_inliers: int
    rmse_inliers: float
    iters: int


def ransac_se3(
    src_corr: Points,
    dst_corr: Points,
    *,
    params: RANSACParams = RANSACParams(),
) -> RANSACResult:
    """
    RANSAC over 3-point minimal samples to estimate an SE(3) transform.

    Behavior:
    - repeatedly sample 3 correspondences, fit SE(3), count inliers.
    - refit using all inliers when an improved model is found.
    - adaptively reduces the required iterations based on best inlier ratio.

    Input requirement:
    - correspondences must have a non-trivial inlier fraction.
    """
    src = _as_f64(src_corr)
    dst = _as_f64(dst_corr)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src_corr and dst_corr must have same shape (K,3).")
    K = src.shape[0]
    if K < 3:
        raise ValueError("Need at least 3 correspondences for SE(3) RANSAC.")

    rng = np.random.default_rng(params.seed)

    best_T = np.eye(4, dtype=np.float64)
    best_inliers = np.zeros(K, dtype=bool)
    best_cnt = 0
    best_rmse = np.inf

    max_iter_adapt = params.max_iter
    it = 0

    def nondegenerate(tri: Points) -> bool:
        # Reject nearly collinear triples; they do not constrain 3D rotation well.
        a, b, c = tri
        return float(np.linalg.norm(np.cross(b - a, c - a))) > 1e-8

    while it < params.max_iter and it < max_iter_adapt:
        it += 1

        idx = rng.choice(K, size=3, replace=False)
        if not nondegenerate(src[idx]):
            continue

        T = kabsch_se3(src[idx], dst[idx])
        src_t = apply_transform(T, src)
        d = np.linalg.norm(src_t - dst, axis=1)

        inl = d < params.threshold
        cnt = int(inl.sum())
        if cnt <= best_cnt:
            continue

        best_cnt = cnt
        best_inliers = inl

        # Refit to all inliers (standard RANSAC refinement).
        best_T = kabsch_se3(src[inl], dst[inl])
        di = np.linalg.norm(apply_transform(best_T, src[inl]) - dst[inl], axis=1)
        best_rmse = float(np.sqrt(np.mean(di * di))) if di.size else np.inf

        # Adaptive stopping: expected number of iterations to reach confidence.
        w = best_cnt / K
        p_no_outliers = 1.0 - (w ** 3)  # minimal sample size is 3
        p_no_outliers = float(np.clip(p_no_outliers, 1e-12, 1.0 - 1e-12))
        max_iter_adapt = int(np.ceil(np.log(1.0 - params.confidence) / np.log(p_no_outliers)))

    return RANSACResult(
        T=best_T,
        inliers=best_inliers,
        num_inliers=best_cnt,
        rmse_inliers=best_rmse,
        iters=it,
    )


# ---------------------------------------------------------------------------
# Baseline correspondences via nearest neighbors
# Principle: keep "demo helpers" clearly labeled; do not imply this is robust.
# ---------------------------------------------------------------------------

def correspondences_via_nn(source: Points, target: Points, *, max_pairs: int = 3000) -> tuple[Points, Points]:
    """
    Build putative correspondences by nearest neighbor in Euclidean space.

    This is only reliable when the point clouds are already roughly aligned.
    For global registration from scratch, use features/descriptors.
    """
    src = _as_f64(source)
    dst = _as_f64(target)

    tree = cKDTree(dst)
    d, idx = tree.query(src, k=1)

    order = np.argsort(d)[: min(max_pairs, d.size)]
    return src[order], dst[np.asarray(idx, dtype=np.int64)[order]]


# ---------------------------------------------------------------------------
# Synthetic test harness
# Principle: deterministic, parameterized, and separated from core solvers.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SynthConfig:
    """
    Defines a synthetic experiment regime for quick sanity checks.
    """
    name: str
    N: int = 1500
    noise: float = 0.01
    outlier_ratio: float = 0.2
    overlap: float = 1.0

    ransac_thresh: float = 0.06
    ransac_iter: int = 3000
    max_pairs: int = 3000

    icp_mode: ICPMode = ICPMode.POINT_TO_POINT
    icp_iter: int = 80
    trim_ratio: float = 0.3

    init_rot_sigma: float = 0.06
    init_trans_sigma: float = 0.06

    true_corr_K: int = 700
    wrong_match_ratio: float = 0.7
    true_ransac_thresh: float = 0.035
    true_ransac_iter: int = 4000


def _random_rotation(rng: np.random.Generator) -> Mat3:
    """
    Generate a random proper rotation matrix (QR-based).
    """
    M = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return _as_f64(Q)


def synth_pair(rng: np.random.Generator, cfg: SynthConfig) -> tuple[Points, Points, Mat4, Points]:
    """
    Create a synthetic registration problem.

    Returns:
      - src_obs: noisy source points plus optional extra outlier points
      - tgt: target point set (possibly partial overlap)
      - T_gt: ground-truth transform mapping source -> target
      - src_inlier: source points that correspond to target points (no extra outliers)
    """
    tgt_full = rng.uniform(-1.0, 1.0, size=(cfg.N, 3)).astype(np.float64)

    if cfg.overlap < 1.0:
        M = int(np.floor(cfg.overlap * cfg.N))
        keep = rng.choice(cfg.N, size=M, replace=False)
        tgt = tgt_full[keep]
    else:
        tgt = tgt_full

    R = _random_rotation(rng)
    t = rng.uniform(-0.5, 0.5, size=(3,)).astype(np.float64)
    T_gt = make_transform(R, t)

    # Source is created by applying the inverse transform to the target.
    src_clean = apply_transform(np.linalg.inv(T_gt), tgt)
    src_inlier = (src_clean + rng.normal(scale=cfg.noise, size=src_clean.shape)).astype(np.float64)

    src_obs = src_inlier
    if cfg.outlier_ratio > 0.0:
        outN = int(np.floor(cfg.outlier_ratio * src_inlier.shape[0]))
        out = rng.uniform(-3.0, 3.0, size=(outN, 3)).astype(np.float64)
        src_obs = np.vstack([src_inlier, out])

    return src_obs, tgt, T_gt, src_inlier


def perturb_pose(rng: np.random.Generator, T_gt: Mat4, *, rot_sigma: float, trans_sigma: float) -> Mat4:
    """
    Create an initial guess by perturbing the ground truth.

    This simulates having a coarse pose estimate from another system (e.g., odometry).
    """
    w = rng.normal(scale=rot_sigma, size=(3,)).astype(np.float64)
    dR = rodrigues(w)
    dt = rng.normal(scale=trans_sigma, size=(3,)).astype(np.float64)
    return make_transform(dR, dt) @ _as_f64(T_gt)


def true_correspondences_with_wrong_matches(
    rng: np.random.Generator,
    src_inlier: Points,
    tgt: Points,
    *,
    K: int,
    wrong_ratio: float,
) -> tuple[Points, Points]:
    """
    Simulate feature matching:
    - pick K true correspondences
    - replace a fraction with wrong matches to model outlier correspondences
    """
    N = tgt.shape[0]
    idx = rng.choice(N, size=min(K, N), replace=False)

    src_corr = src_inlier[idx]
    dst_corr = tgt[idx].copy()

    wrong = int(np.floor(wrong_ratio * idx.size))
    if wrong > 0:
        wrong_idx = rng.choice(idx.size, size=wrong, replace=False)
        dst_corr[wrong_idx] = rng.uniform(-1.0, 1.0, size=(wrong, 3)).astype(np.float64)

    return src_corr, dst_corr


def run_one(cfg: SynthConfig, *, trial_seed: int) -> dict[str, float | int]:
    """
    Run a single trial and return scalar metrics for easy aggregation.
    """
    rng = np.random.default_rng(trial_seed)

    src_obs, tgt, T_gt, src_inlier = synth_pair(rng, cfg)
    icp_params = ICPParams(max_iter=cfg.icp_iter, tol=1e-7, trim_ratio=cfg.trim_ratio, mode=cfg.icp_mode)

    # A0: NN-based correspondences (raw frame) -> RANSAC -> ICP
    sc0, dc0 = correspondences_via_nn(src_obs, tgt, max_pairs=cfg.max_pairs)
    r0 = ransac_se3(
        sc0, dc0,
        params=RANSACParams(threshold=cfg.ransac_thresh, max_iter=cfg.ransac_iter, seed=trial_seed),
    )
    a0 = icp(src_obs, tgt, init_T=r0.T, params=icp_params)
    a0_rot, a0_trans = pose_error(a0.T, T_gt)

    # B: ICP-only with a decent initial guess
    T_init = perturb_pose(rng, T_gt, rot_sigma=cfg.init_rot_sigma, trans_sigma=cfg.init_trans_sigma)
    b = icp(src_obs, tgt, init_T=T_init, params=icp_params)
    b_rot, b_trans = pose_error(b.T, T_gt)

    # C: RANSAC from meaningful correspondences (simulated feature matches) -> ICP
    sc, dc = true_correspondences_with_wrong_matches(
        rng, src_inlier, tgt, K=cfg.true_corr_K, wrong_ratio=cfg.wrong_match_ratio
    )
    rC = ransac_se3(
        sc, dc,
        params=RANSACParams(threshold=cfg.true_ransac_thresh, max_iter=cfg.true_ransac_iter, seed=trial_seed + 999),
    )
    c = icp(src_obs, tgt, init_T=rC.T, params=icp_params)
    c_rot, c_trans = pose_error(c.T, T_gt)

    rot_ok = 0.05
    trans_ok = 0.05

    return {
        "A0_success": int((a0_rot < rot_ok) and (a0_trans < trans_ok)),
        "A0_rot": a0_rot,
        "A0_trans": a0_trans,
        "A0_conv": int(a0.converged),

        "B_success": int((b_rot < rot_ok) and (b_trans < trans_ok)),
        "B_rot": b_rot,
        "B_trans": b_trans,
        "B_conv": int(b.converged),

        "C_success": int((c_rot < rot_ok) and (c_trans < trans_ok)),
        "C_rot": c_rot,
        "C_trans": c_trans,
        "C_conv": int(c.converged),
        "C_inliers": int(rC.num_inliers),
    }


def run_suite(configs: list[SynthConfig], *, trials: int = 5, seed0: int = 10) -> list[dict[str, object]]:
    """
    Run multiple configurations and trials.

    Returns:
      A list of dicts (row-like records) to keep the module lightweight and flexible.
    """
    rows: list[dict[str, object]] = []
    for cfg in configs:
        for t in range(trials):
            # Deterministic seed per (config, trial) for reproducibility.
            trial_seed = seed0 + (hash(cfg.name) & 0xFFFF) + 1000 * t
            rows.append({"config": cfg.name, "trial": t, **run_one(cfg, trial_seed=trial_seed)})
    return rows


def summarize(rows: list[dict[str, object]]) -> str:
    """
    Produce a chat-friendly markdown table from trial records.

    Principle:
    - Keep output formatting localized; do not mix printing with solver logic.
    """
    by: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by.setdefault(str(r["config"]), []).append(r)

    def mean(xs: list[float]) -> float:
        return float(np.mean(np.asarray(xs, dtype=np.float64)))

    def median(xs: list[float]) -> float:
        return float(np.median(np.asarray(xs, dtype=np.float64)))

    header = (
        "| Config | Trials | A0 success | B success | C success | "
        "A0 med rot | B med rot | C med rot | C med inliers |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]

    for name, rs in by.items():
        trials = len(rs)
        a0s = [float(r["A0_success"]) for r in rs]
        bs = [float(r["B_success"]) for r in rs]
        cs = [float(r["C_success"]) for r in rs]
        a0rot = [float(r["A0_rot"]) for r in rs]
        brot = [float(r["B_rot"]) for r in rs]
        crot = [float(r["C_rot"]) for r in rs]
        cinl = [float(r["C_inliers"]) for r in rs]

        lines.append(
            f"| {name} | {trials} | {mean(a0s):.2f} | {mean(bs):.2f} | {mean(cs):.2f} | "
            f"{median(a0rot):.4f} | {median(brot):.4f} | {median(crot):.4f} | {median(cinl):.0f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# Principle: keep side effects here; core logic stays importable and testable.
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description="ICP + RANSAC 3D registration demo")
    ap.add_argument("--trials", type=int, default=5, help="number of trials per configuration")
    args = ap.parse_args()

    configs = [
        SynthConfig(
            name="moderate: noise 0.01, outliers 20%, overlap 100%",
            N=1200, noise=0.01, outlier_ratio=0.2, overlap=1.0,
            ransac_thresh=0.06, ransac_iter=2000, max_pairs=2500,
            icp_iter=60, trim_ratio=0.3,
            init_rot_sigma=0.05, init_trans_sigma=0.05,
            true_corr_K=600, wrong_match_ratio=0.7, true_ransac_thresh=0.03, true_ransac_iter=3000,
        ),
        SynthConfig(
            name="partial: noise 0.01, outliers 20%, overlap 60%",
            N=1600, noise=0.01, outlier_ratio=0.2, overlap=0.6,
            ransac_thresh=0.07, ransac_iter=2500, max_pairs=3000,
            icp_iter=90, trim_ratio=0.4,
            init_rot_sigma=0.05, init_trans_sigma=0.05,
            true_corr_K=700, wrong_match_ratio=0.7, true_ransac_thresh=0.035, true_ransac_iter=4000,
        ),
        SynthConfig(
            name="hard: noise 0.02, outliers 40%, overlap 100%",
            N=1200, noise=0.02, outlier_ratio=0.4, overlap=1.0,
            ransac_thresh=0.08, ransac_iter=2500, max_pairs=3000,
            icp_iter=80, trim_ratio=0.4,
            init_rot_sigma=0.06, init_trans_sigma=0.06,
            true_corr_K=600, wrong_match_ratio=0.75, true_ransac_thresh=0.045, true_ransac_iter=4500,
        ),
    ]

    rows = run_suite(configs, trials=args.trials, seed0=21)
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
