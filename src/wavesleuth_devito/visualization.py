"""Matplotlib visualizations for worlds, runs, reconstructions, and uncertainty."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .geometry import receiver_coordinates, source_coordinates
from .io import ensure_parent, load_json, load_run_npz, world_from_run
from .world import anomaly_kind, velocity_model_from_world


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _domain_extent(world: dict[str, Any]) -> list[float]:
    return [0.0, float(world["grid"]["extent_x"]), 0.0, float(world["grid"]["extent_z"])]


def _safe_center_extent(xs: np.ndarray, zs: np.ndarray, world: dict[str, Any]) -> list[float]:
    """Return an imshow extent that never has identical min/max limits."""
    extent_x = float(world["grid"]["extent_x"])
    extent_z = float(world["grid"]["extent_z"])
    if xs.size == 0 or zs.size == 0:
        return _domain_extent(world)
    x0 = float(np.nanmin(xs))
    x1 = float(np.nanmax(xs))
    z0 = float(np.nanmin(zs))
    z1 = float(np.nanmax(zs))

    if not np.isfinite([x0, x1, z0, z1]).all():
        return _domain_extent(world)

    if abs(x1 - x0) <= 1.0e-12:
        pad = max(1.0e-6, 0.025 * extent_x)
        x0 = max(0.0, x0 - pad)
        x1 = min(extent_x, x1 + pad)
        if abs(x1 - x0) <= 1.0e-12:
            x1 = x0 + pad
    if abs(z1 - z0) <= 1.0e-12:
        pad = max(1.0e-6, 0.025 * extent_z)
        z0 = max(0.0, z0 - pad)
        z1 = min(extent_z, z1 + pad)
        if abs(z1 - z0) <= 1.0e-12:
            z1 = z0 + pad
    return [x0, x1, z0, z1]


def _overlay_acquisition(ax: Any, world: dict[str, Any]) -> None:
    src = source_coordinates(world)
    rec = receiver_coordinates(world)
    ax.scatter(src[:, 0], src[:, 1], marker="*", s=120, label="sources")
    ax.scatter(rec[:, 0], rec[:, 1], marker="v", s=60, label="receivers")
    ax.legend(loc="upper right", fontsize=8)


def _overlay_anomaly(ax: Any, world: dict[str, Any], *, label: str = "true anomaly") -> None:
    plt = _plt()
    anomaly = world["medium"]["anomaly"]
    kind = anomaly_kind(world)
    if kind == "circle":
        ax.add_patch(plt.Circle((float(anomaly["center_x"]), float(anomaly["center_z"])), float(anomaly["radius"]), fill=False, linewidth=2.0, label=label))
    elif kind == "rectangle":
        width = float(anomaly["width"])
        height = float(anomaly["height"])
        lower_left = (float(anomaly["center_x"]) - width / 2.0, float(anomaly["center_z"]) - height / 2.0)
        ax.add_patch(plt.Rectangle(lower_left, width, height, fill=False, linewidth=2.0, label=label))
    elif kind == "blobs":
        for idx, blob in enumerate(anomaly["blobs"]):
            ax.add_patch(plt.Circle((float(blob["center_x"]), float(blob["center_z"])), float(blob["radius"]), fill=False, linewidth=1.5, label=label if idx == 0 else None))
    elif kind == "layered":
        for layer in anomaly["layers"]:
            ax.axhline(float(layer["z_min"]), linewidth=1.0)
            ax.axhline(float(layer["z_max"]), linewidth=1.0)


def visualize_world(world_or_path: dict[str, Any] | str | Path, out_path: str | Path) -> Path:
    """Plot a velocity model and overlay acquisition geometry."""
    world = load_json(world_or_path) if isinstance(world_or_path, (str, Path)) else world_or_path
    velocity_model = velocity_model_from_world(world)
    plt = _plt()
    out = ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(velocity_model.T, origin="lower", extent=_domain_extent(world), aspect="equal")
    fig.colorbar(image, ax=ax, label="velocity")
    _overlay_anomaly(ax, world)
    _overlay_acquisition(ax, world)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"WaveSleuth world: {world.get('name', 'unnamed')}")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _trace_image(ax: Any, fig: Any, traces: np.ndarray, time: np.ndarray, *, title: str) -> None:
    vmax = float(np.percentile(np.abs(traces), 99.0)) if np.any(traces) else 1.0
    image = ax.imshow(traces.T, origin="lower", aspect="auto", extent=[float(time[0]), float(time[-1]), -0.5, traces.shape[1] - 0.5], vmin=-vmax, vmax=vmax)
    fig.colorbar(image, ax=ax, label="pressure")
    ax.set_xlabel("time")
    ax.set_ylabel("receiver")
    ax.set_title(title)


def visualize_run(run_path: str | Path, out_path: str | Path) -> Path:
    """Plot receiver traces from a `.npz` run."""
    run = load_run_npz(run_path)
    world = world_from_run(run)
    traces = np.asarray(run["receiver_traces"], dtype=np.float32)
    time = np.asarray(run["time"], dtype=np.float32)
    plt = _plt()
    out = ensure_parent(out_path)
    if traces.ndim == 2:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        _trace_image(ax, fig, traces, time, title=f"Receiver traces: {world.get('name', Path(run_path).stem)}")
    elif traces.ndim == 3:
        nshot = traces.shape[0]
        fig, axes = plt.subplots(nshot, 1, figsize=(7.2, max(3.0, 2.6 * nshot)), squeeze=False)
        for ishot in range(nshot):
            _trace_image(axes[ishot, 0], fig, traces[ishot], time, title=f"Shot {ishot}: receiver traces")
        fig.suptitle(f"Sequential-shot run: {world.get('name', Path(run_path).stem)}")
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.text(0.5, 0.5, f"Unsupported trace shape {traces.shape}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _latest_grid_for_reconstruction(reconstruction: dict[str, Any]) -> tuple[list[float], list[float], list[list[float | None]]]:
    levels = reconstruction.get("search_levels")
    if isinstance(levels, list) and levels:
        level = levels[-1]
        return level.get("xs", []), level.get("zs", []), level.get("mismatch_map", [])
    grid = reconstruction.get("candidate_grid", {})
    return grid.get("xs", []), grid.get("zs", []), reconstruction.get("mismatch_map", [])


def visualize_reconstruction(reconstruction_or_path: dict[str, Any] | str | Path, out_path: str | Path) -> Path:
    """Plot true/predicted anomalies and the candidate mismatch map."""
    reconstruction = load_json(reconstruction_or_path) if isinstance(reconstruction_or_path, (str, Path)) else reconstruction_or_path
    world = reconstruction.get("world")
    if not isinstance(world, dict):
        raise ValueError("Reconstruction does not contain embedded world metadata.")
    velocity_model = velocity_model_from_world(world)
    best = reconstruction.get("best_candidate", {})
    true = reconstruction.get("true_center", {})
    objective = reconstruction.get("objective", {})
    plt = _plt()
    out = ensure_parent(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    ax0 = axes[0]
    image0 = ax0.imshow(velocity_model.T, origin="lower", extent=_domain_extent(world), aspect="equal")
    fig.colorbar(image0, ax=ax0, label="velocity")
    _overlay_anomaly(ax0, world, label="true")
    if best:
        label = f"reconstructed r={float(best.get('radius', 0.0)):.3f}"
        ax0.add_patch(plt.Circle((float(best["center_x"]), float(best["center_z"])), float(best["radius"]), fill=False, linestyle="--", linewidth=2.0, label=label))
    ax0.set_title("True and reconstructed anomaly")
    ax0.set_xlabel("x")
    ax0.set_ylabel("z")
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    xs_raw, zs_raw, mismatch_raw = _latest_grid_for_reconstruction(reconstruction)
    if mismatch_raw:
        mismatch = np.asarray([[np.nan if value is None else float(value) for value in row] for row in mismatch_raw], dtype=np.float64)
        xs = np.asarray(xs_raw, dtype=float)
        zs = np.asarray(zs_raw, dtype=float)
        extent = _safe_center_extent(xs, zs, world) if xs.size and zs.size else _domain_extent(world)
        image1 = ax1.imshow(mismatch, origin="lower", aspect="auto", extent=extent)
        fig.colorbar(image1, ax=ax1, label="best mismatch at center")
        if true:
            ax1.scatter([float(true["center_x"])], [float(true["center_z"])], marker="o", s=80, label="true center")
        if best:
            ax1.scatter([float(best["center_x"])], [float(best["center_z"])], marker="x", s=90, label="best")
        ax1.legend(loc="upper right", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No mismatch map", ha="center", va="center")
    mode = objective.get("mismatch_mode", "unknown")
    metric = objective.get("metric", "unknown")
    ax1.set_title(f"Candidate mismatch map\n{mode}, {metric}")
    ax1.set_xlabel("candidate center x")
    ax1.set_ylabel("candidate center z")

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _candidate_center_probabilities(reconstruction: dict[str, Any], temperature: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = list(reconstruction.get("candidates", []))
    if not candidates:
        return np.asarray([]), np.asarray([]), np.asarray([[]])
    xs = np.asarray(sorted({round(float(c["center_x"]), 8) for c in candidates}), dtype=float)
    zs = np.asarray(sorted({round(float(c["center_z"]), 8) for c in candidates}), dtype=float)
    grid = np.zeros((len(zs), len(xs)), dtype=float)
    x_index = {round(float(x), 8): i for i, x in enumerate(xs)}
    z_index = {round(float(z), 8): i for i, z in enumerate(zs)}
    mismatches = np.asarray([float(c["mismatch"]) for c in candidates], dtype=float)
    finite = np.isfinite(mismatches)
    if not bool(np.any(finite)):
        return xs, zs, grid
    if temperature is None:
        temp = float((reconstruction.get("uncertainty") or {}).get("temperature", 0.0) or 0.0)
    else:
        temp = float(temperature)
    if temp <= 0.0:
        temp = max(float(np.nanpercentile(mismatches[finite], 75.0) - np.nanmin(mismatches[finite])), 1.0e-9)
    logits = -(mismatches - np.nanmin(mismatches[finite])) / temp
    logits[~finite] = -np.inf
    logits -= float(np.nanmax(logits))
    raw = np.exp(logits)
    probs = raw / max(float(raw.sum()), 1.0e-12)
    for c, p in zip(candidates, probs):
        grid[z_index[round(float(c["center_z"]), 8)], x_index[round(float(c["center_x"]), 8)]] += float(p)
    return xs, zs, grid


def _uncertainty_summary_for_title(reconstruction: dict[str, Any], temperature: float | None) -> dict[str, Any]:
    summary = reconstruction.get("uncertainty") or {}
    if temperature is None and summary.get("effective_candidates") is not None:
        return summary
    try:
        from .uncertainty import candidate_probabilities

        return candidate_probabilities(reconstruction, temperature=temperature)
    except Exception:
        return summary if isinstance(summary, dict) else {}


def visualize_uncertainty(reconstruction_or_path: dict[str, Any] | str | Path, out_path: str | Path, *, temperature: float | None = None) -> Path:
    """Plot a pseudo-probability map from candidate mismatches."""
    reconstruction = load_json(reconstruction_or_path) if isinstance(reconstruction_or_path, (str, Path)) else reconstruction_or_path
    world = reconstruction.get("world")
    if not isinstance(world, dict):
        raise ValueError("Reconstruction does not contain embedded world metadata.")
    best = reconstruction.get("best_candidate", {})
    true = reconstruction.get("true_center", {})
    uncertainty = _uncertainty_summary_for_title(reconstruction, temperature)
    plt = _plt()
    out = ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    xs, zs, grid = _candidate_center_probabilities(reconstruction, temperature=temperature)
    if xs.size and zs.size:
        extent = _safe_center_extent(xs, zs, world)
        image = ax.imshow(grid, origin="lower", aspect="auto", extent=extent)
        fig.colorbar(image, ax=ax, label="pseudo-probability by center")
    else:
        ax.text(0.5, 0.5, "No candidate uncertainty available", ha="center", va="center")
    if true:
        ax.scatter([float(true["center_x"])], [float(true["center_z"])], marker="o", s=85, label="true center")
    if best:
        ax.scatter([float(best["center_x"])], [float(best["center_z"])], marker="x", s=95, label="best")
    ax.set_xlim(0.0, float(world["grid"]["extent_x"]))
    ax.set_ylim(0.0, float(world["grid"]["extent_z"]))
    ax.set_xlabel("candidate center x")
    ax.set_ylabel("candidate center z")
    effective = float(uncertainty.get("effective_candidates", 0.0) or 0.0)
    center_effective = float(uncertainty.get("center_effective_candidates", 0.0) or 0.0)
    search = reconstruction.get("search", {})
    strategy = str(search.get("search_strategy", "unknown")) if isinstance(search, dict) else "unknown"
    if strategy == "staged":
        title = (
            "Uncertainty by candidate center, staged search\n"
            f"entropy={float(uncertainty.get('normalized_entropy', 0.0) or 0.0):.3f}, "
            f"center-effective={center_effective:.1f}, raw-effective={effective:.1f}"
        )
    else:
        title = (
            "Uncertainty from mismatch surface\n"
            f"entropy={float(uncertainty.get('normalized_entropy', 0.0) or 0.0):.3f}, "
            f"effective={effective:.1f}, centers={center_effective:.1f}"
        )
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
