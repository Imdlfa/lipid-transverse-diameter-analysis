#!/usr/bin/env python3
from __future__ import annotations
import itertools
import time
from pathlib import Path
import numpy as np

# ===== User Parameters =====
F_TOP = "md_zedian.tpr"  # Topology file
F_TRJ = "md_zedian.xtc"  # Trajectory file
SEL_RN = ["MOL"]  # Selected resnames
SEL_RI = None  # Selected resids list
SEL_RR = [(1, 110)]  # CE resid range; TG = [(111, 220)]
FR_BEG = 0  # Start frame index
FR_END = -1  # End frame index (-1 for last)
FR_STEP = 5  # Frame stride (~0.1 ns with DT_NS)
DT_NS = 0.02  # nstxout-compressed=10000, dt=0.002 ps -> 0.02 ns/frame
UNW = True # Unwrap residues per frame
B_D = 100  # Number of diameter bins
B_T = 100  # Number of time bins
# ===== Output configs =====
OUT = "CE_diameter_out"  # Output directory; TG -> TG_diameter_out
O_RAW = "raw.csv"  # Raw per-molecule CSV
O_GRID = "grid.csv"  # Grid/probability CSV
O_MEAN = "mean.csv"  # Mean series CSV
FIG_FMT = "svg"  # Unified plot format: png, svg, pdf, eps
O_3D = "dist3d"  # 3D distribution plot basename (extension from FIG_FMT)
O_MEAN_FIG = "mean"  # Mean series plot basename (extension from FIG_FMT)
DPI = 300  # Raster DPI (png/pdf); ignored for svg
DO_CALC = 1  # Enable calculation
DO_3D = 1  # Enable 3D plotting
DO_MEAN = 1  # Enable mean time-series
X_CLIP = 1  # Clip x-range for 3D plot
X_MIN = 0.0  # X minimum (Å)
X_MAX = 30.0  # X maximum (Å)
SM = 1  # Enable smoothing
SM_D = 3.0  # Smoothing sigma on diameter axis
SM_T = 1.0  # Smoothing sigma on time axis
SM_N = 1  # Number of smoothing passes
CMAP_3D = "inferno"  # 3D surface colormap: "viridis" (default) or "inferno"
# ===========================

def _plot_path(basename: str) -> Path:
    fmt = str(FIG_FMT).lower().lstrip(".")
    p = Path(basename)
    return p.with_suffix(f".{fmt}") if p.suffix else p.with_suffix(f".{fmt}")
def _save_fig(fig, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    kw: dict = {"bbox_inches": "tight"}
    if out_path.suffix.lower() != ".svg":
        kw["dpi"] = DPI
    fig.savefig(out_path, **kw); plt.close(fig)
def _log(msg: str) -> None:
    print(msg, flush=True)
def _pbar(done: int, total: int, width: int = 30) -> None:
    total_safe = max(1, int(total)); done_clamped = min(max(int(done), 0), total_safe)
    frac = done_clamped / total_safe; filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rProgress [{bar}] {done_clamped:>6}/{total_safe:<6} ({frac * 100:5.1f}%)", end="", flush=True)
def _sstop(stop: int, n_frames: int) -> int:
    return n_frames if stop < 0 else min(stop, n_frames)
def _nfrm(start: int, stop: int, stride: int) -> int:
    return 0 if stop <= start or stride <= 0 else (stop - start + stride - 1) // stride
def _ridset() -> set[int] | None:
    selected: set[int] = set(); has_filter = False
    if SEL_RI is not None:
        selected.update(int(x) for x in SEL_RI); has_filter = True
    if SEL_RR is not None:
        has_filter = True
        for pair in SEL_RR:
            if len(pair) != 2: raise ValueError(f"Invalid SEL_RR element: {pair!r}")
            a, b = int(pair[0]), int(pair[1]); lo, hi = (a, b) if a <= b else (b, a)
            selected.update(range(lo, hi + 1))
    return selected if has_filter else None
def _gk1(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0: return np.array([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(3.0 * sigma_bins))); x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma_bins * sigma_bins)); k /= k.sum(); return k
def _sm2(H: np.ndarray, sigma_x: float, sigma_y: float, passes: int) -> np.ndarray:
    def conv_same_len(v: np.ndarray, k: np.ndarray) -> np.ndarray:
        n, m = v.shape[0], k.shape[0]
        if n == 0 or m <= 1: return v.copy()
        pad_left = m // 2; pad_right = m - 1 - pad_left
        return np.convolve(np.pad(v, (pad_left, pad_right), mode="edge"), k, mode="valid")
    out = np.asarray(H, dtype=np.float64).copy()
    if passes <= 0 or (sigma_x <= 0 and sigma_y <= 0): return out
    kx, ky = _gk1(sigma_x), _gk1(sigma_y)
    for _ in range(passes):
        if sigma_x > 0: out = np.apply_along_axis(lambda v: conv_same_len(v, kx), axis=0, arr=out)
        if sigma_y > 0: out = np.apply_along_axis(lambda v: conv_same_len(v, ky), axis=1, arr=out)
    return out
def _hull2(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= 2: return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p.copy())
    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p.copy())
    hull = lower[:-1] + upper[:-1]
    return np.stack(hull, axis=0) if hull else pts
def _cc3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    ax, ay, bx, by, cx, cy = float(a[0]), float(a[1]), float(b[0]), float(b[1]), float(c[0]), float(c[1])
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-14: return None, None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    center = np.array([ux, uy], dtype=np.float64)
    return center, float(np.linalg.norm(center - a))
def _mec2(points: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=np.float64); n = pts.shape[0]
    if n == 0: raise ValueError("Empty point set")
    if n == 1: return pts[0].copy(), 0.0
    if n == 2:
        c = (pts[0] + pts[1]) / 2.0
        return c, float(np.linalg.norm(pts[0] - c))
    def all_inside(center: np.ndarray, r: float, eps: float = 1e-9) -> bool:
        return bool(np.all(np.linalg.norm(pts - center, axis=1) <= r + eps))
    cand = _hull2(pts); m = cand.shape[0]
    best_r = np.inf; best_c: np.ndarray | None = None
    for i, j in itertools.combinations(range(m), 2):
        p1, p2 = cand[i], cand[j]; c = (p1 + p2) / 2.0; r = float(np.linalg.norm(p1 - c))
        if r < best_r and all_inside(c, r): best_r, best_c = r, c
    for i, j, k in itertools.combinations(range(m), 3):
        c, r = _cc3(cand[i], cand[j], cand[k])
        if c is not None and r is not None and r < best_r and all_inside(c, r): best_r, best_c = r, c
    if best_c is None or not np.isfinite(best_r):
        max_d = -1.0; best_c = (pts[0] + pts[1]) / 2.0; best_r = 0.0
        for i, j in itertools.combinations(range(n), 2):
            c = (pts[i] + pts[j]) / 2.0; r = float(np.linalg.norm(pts[i] - c)); d = float(np.linalg.norm(pts[i] - pts[j]))
            if d > max_d: max_d, best_c, best_r = d, c, r
    return best_c, float(best_r)
def _dproj(positions_angstrom: np.ndarray) -> float:
    """Projected min-circle diameter. MDAnalysis positions are Å; do not apply nm→Å again."""
    if positions_angstrom.size == 0: return 0.0
    pos = np.asarray(positions_angstrom, dtype=np.float64); n = pos.shape[0]
    if n == 1: return 0.0
    c = pos.mean(axis=0); X = pos - c; cov = (X.T @ X) / n
    evals, evecs = np.linalg.eigh(cov); u = evecs[:, int(np.argmax(evals))]
    un = np.linalg.norm(u); u = np.array([0.0, 0.0, 1.0], dtype=np.float64) if un < 1e-14 else u / un
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(u, tmp)) > 0.9: tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = np.cross(u, tmp); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1); e2 /= np.linalg.norm(e2)
    _, r = _mec2(np.column_stack((X @ e1, X @ e2)))
    return float(2.0 * r)
def _calc_raw(u, start_eff: int, stop_eff: int, residues) -> tuple[list[tuple[int, float, str, int, float]], np.ndarray, np.ndarray]:
    ag_filtered = residues[0].atoms
    for r in residues[1:]: ag_filtered = ag_filtered + r.atoms
    raw_rows: list[tuple[int, float, str, int, float]] = []; diameters_list: list[float] = []; times_list: list[float] = []
    total_frames = _nfrm(start_eff, stop_eff, FR_STEP)
    update_every = max(1, total_frames // 100)
    _pbar(0, total_frames)
    for i, ts in enumerate(u.trajectory[start_eff:stop_eff:FR_STEP], start=1):
        frame_idx = ts.frame; time_ns = frame_idx * float(DT_NS)
        if UNW: ag_filtered.unwrap(compound="residues")
        for res in residues:
            d_ang = _dproj(res.atoms.positions.copy())
            raw_rows.append((frame_idx, time_ns, res.resname, int(res.resid), d_ang))
            diameters_list.append(d_ang); times_list.append(time_ns)
        if i % update_every == 0 or i == total_frames: _pbar(i, total_frames)
    print()
    return raw_rows, np.asarray(diameters_list, dtype=np.float64), np.asarray(times_list, dtype=np.float64)
def _save_raw(raw_rows: list[tuple[int, float, str, int, float]], raw_path: Path) -> None:
    with open(raw_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("frame_index,time_ns,resname,resid,diameter_angstrom\n")
        for row in raw_rows: f.write(f"{row[0]},{row[1]:.6f},{row[2]},{row[3]},{row[4]:.6f}\n")
def _read_rawtbl(raw_path: Path) -> np.ndarray:
    if not raw_path.exists(): raise RuntimeError(f"Missing raw CSV: {raw_path}")
    data = np.genfromtxt(raw_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0: raise RuntimeError(f"Empty raw CSV: {raw_path}")
    return np.array([data], dtype=data.dtype) if data.ndim == 0 else data
def _read_raw(raw_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = _read_rawtbl(raw_path)
    return np.asarray(data["diameter_angstrom"], dtype=np.float64), np.asarray(data["time_ns"], dtype=np.float64)
def _mean_series(raw_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = np.asarray(raw_table["frame_index"], dtype=np.int64); time_ns = np.asarray(raw_table["time_ns"], dtype=np.float64); d = np.asarray(raw_table["diameter_angstrom"], dtype=np.float64)
    if frame.size == 0: raise RuntimeError("No data for mean time series")
    order = np.argsort(frame, kind="mergesort"); f_sorted = frame[order]; t_sorted = time_ns[order]; d_sorted = d[order]
    uniq_frame, start_idx = np.unique(f_sorted, return_index=True); end_idx = np.append(start_idx[1:], f_sorted.size)
    counts = end_idx - start_idx; mean_d = np.add.reduceat(d_sorted, start_idx) / counts; uniq_time = t_sorted[start_idx]
    return uniq_frame, uniq_time, mean_d, counts
def _save_mean(frame_idx: np.ndarray, time_ns: np.ndarray, mean_d: np.ndarray, counts: np.ndarray, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("frame_index,time_ns,diameter_mean_angstrom,n_molecules\n")
        for i in range(frame_idx.size): f.write(f"{int(frame_idx[i])},{time_ns[i]:.8f},{mean_d[i]:.8f},{int(counts[i])}\n")
def _plot_mean(time_ns: np.ndarray, mean_d: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 4.8)); ax = fig.add_subplot(111)
    ax.plot(time_ns, mean_d, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Time / ns"); ax.set_ylabel("Mean diameter / Å"); ax.set_title("Mean diameter vs time")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    _save_fig(fig, out_path)
def _hist2(diameters: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if diameters.size == 0 or times.size == 0: raise RuntimeError("No data for plotting")
    if X_CLIP:
        d_lo, d_hi = float(min(X_MIN, X_MAX)), float(max(X_MIN, X_MAX))
        mask = (diameters >= d_lo) & (diameters <= d_hi)
        if not np.any(mask): raise RuntimeError(f"No data in x-range [{d_lo}, {d_hi}] Å")
        diameters, times = diameters[mask], times[mask]; d_min, d_max = d_lo, d_hi
    else:
        d_min, d_max = float(diameters.min()), float(diameters.max())
    t_min, t_max = float(times.min()), float(times.max())
    if d_max <= d_min: d_max = d_min + 1e-6
    if t_max <= t_min: t_max = t_min + 1e-6
    return np.histogram2d(diameters, times, bins=[B_D, B_T], range=[[d_min, d_max], [t_min, t_max]])
def _norm_t(H_count: np.ndarray) -> np.ndarray:
    H = np.asarray(H_count, dtype=np.float64); col_sum = H.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(H, col_sum, out=np.zeros_like(H, dtype=np.float64), where=col_sum > 0)
def _save_grid(H_count: np.ndarray, H_prob: np.ndarray, H_prob_smooth: np.ndarray, d_edges: np.ndarray, t_edges: np.ndarray, grid_path: Path) -> None:
    d_centers = (d_edges[:-1] + d_edges[1:]) / 2.0; t_centers = (t_edges[:-1] + t_edges[1:]) / 2.0
    with open(grid_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("diameter_angstrom,time_ns,count,prob_t,prob_t_smooth\n")
        for i, dc in enumerate(d_centers):
            for j, tc in enumerate(t_centers): f.write(f"{dc:.8f},{tc:.8f},{int(H_count[i, j])},{H_prob[i, j]:.10e},{H_prob_smooth[i, j]:.10e}\n")
def _plot3(H_prob_plot: np.ndarray, d_edges: np.ndarray, t_edges: np.ndarray, plot_path: Path) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    d_centers = (d_edges[:-1] + d_edges[1:]) / 2.0; t_centers = (t_edges[:-1] + t_edges[1:]) / 2.0
    DD, TT = np.meshgrid(d_centers, t_centers, indexing="xy"); ZZ = H_prob_plot.T
    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(DD, TT, ZZ, cmap=CMAP_3D, edgecolor="none", alpha=0.92, linewidth=0, antialiased=True)
    ax.set_xlabel("Diameter / Å"); ax.set_ylabel("Time / ns"); ax.set_zlabel("Probability")
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, label="Probability"); ax.set_title("Diameter vs time (P(d|t))")
    _save_fig(fig, plot_path)
def main() -> None:
    import MDAnalysis as mda
    t0 = time.perf_counter()
    out_dir = Path(OUT); out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / O_RAW; grid_path = out_dir / O_GRID
    plot_path = out_dir / _plot_path(O_3D); mean_csv_path = out_dir / O_MEAN
    mean_plot_path = out_dir / _plot_path(O_MEAN_FIG)
    diameters: np.ndarray | None = None; times: np.ndarray | None = None
    stages = int(bool(DO_CALC)) + int(bool(DO_3D)) + int(bool(DO_MEAN))
    stage_idx = 0
    if DO_CALC:
        stage_idx += 1; _log(f"[{stage_idx}/{stages}] Calculating raw diameters")
        u = mda.Universe(str(Path(F_TOP)), str(Path(F_TRJ))); n_frames = len(u.trajectory)
        stop_eff = _sstop(FR_END, n_frames); start_eff = max(0, FR_BEG)
        if _nfrm(start_eff, stop_eff, FR_STEP) == 0: raise RuntimeError("No frames to analyze")
        allowed_resnames = set(SEL_RN) if SEL_RN is not None else None; allowed_resids = _ridset()
        residues = [r for r in u.residues if (allowed_resnames is None or r.resname in allowed_resnames) and (allowed_resids is None or int(r.resid) in allowed_resids)]
        if not residues: raise RuntimeError("No residues matched filters")
        raw_rows, diameters, times = _calc_raw(u, start_eff, stop_eff, residues); _save_raw(raw_rows, raw_path)
        _log(f"[{stage_idx}/{stages}] Raw CSV generated: {raw_path.name}")
    if DO_3D:
        stage_idx += 1; _log(f"[{stage_idx}/{stages}] Building 3D distribution")
        if diameters is None or times is None: diameters, times = _read_raw(raw_path)
        H_count, d_edges, t_edges = _hist2(diameters, times); H_prob = _norm_t(H_count)
        H_prob_smooth = _sm2(H_prob, float(SM_D), float(SM_T), int(SM_N)) if SM else H_prob.copy()
        _save_grid(H_count, H_prob, H_prob_smooth, d_edges, t_edges, grid_path); _plot3(H_prob_smooth if SM else H_prob, d_edges, t_edges, plot_path)
        _log(f"[{stage_idx}/{stages}] 3D outputs generated")
    if DO_MEAN:
        stage_idx += 1; _log(f"[{stage_idx}/{stages}] Building mean time series")
        raw_table = _read_rawtbl(raw_path); frame_idx, time_vec, mean_d, counts = _mean_series(raw_table)
        _save_mean(frame_idx, time_vec, mean_d, counts, mean_csv_path); _plot_mean(time_vec, mean_d, mean_plot_path)
        _log(f"[{stage_idx}/{stages}] Mean outputs generated")
    generated_files: list[Path] = []
    for p in (raw_path, grid_path, plot_path, mean_csv_path, mean_plot_path):
        if p.exists(): generated_files.append(p)
    _log(f"Output folder generated: {out_dir.resolve()}")
    for p in generated_files: _log(f"Generated file: {p.name}")
    _log(f"Elapsed: {time.perf_counter() - t0:.2f}s")
    _log("Calculation completed.")
if __name__ == "__main__":
    main()
