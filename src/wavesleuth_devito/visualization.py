"""Matplotlib visualizations for worlds, runs, reconstructions, and uncertainty."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .blind import is_blind_public_world
from .geometry import receiver_coordinates, source_coordinates
from .io import ensure_parent, load_json, load_run_npz, world_from_run
from .world import anomaly_kind, background_velocity_model_from_world, velocity_model_from_world


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


def _add_ellipse_patch(ax: Any, *, center_x: float, center_z: float, radius_x: float, radius_z: float, angle_degrees: float, label: str | None, linestyle: str = "-", linewidth: float = 2.0) -> None:
    from matplotlib.patches import Ellipse

    ax.add_patch(
        Ellipse(
            (float(center_x), float(center_z)),
            width=2.0 * float(radius_x),
            height=2.0 * float(radius_z),
            angle=float(angle_degrees),
            fill=False,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    )


def _add_crack_patch(ax: Any, *, center_x: float, center_z: float, length: float, width: float, angle_degrees: float, label: str | None, linestyle: str = "-", linewidth: float = 2.0) -> None:
    from matplotlib.patches import Polygon

    theta = np.deg2rad(float(angle_degrees))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    half_l = float(length) / 2.0
    half_w = float(width) / 2.0
    local = np.asarray([[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]], dtype=float)
    points = []
    for x_local, z_local in local:
        x = float(center_x) + c * x_local - s * z_local
        z = float(center_z) + s * x_local + c * z_local
        points.append((x, z))
    ax.add_patch(Polygon(points, closed=True, fill=False, linestyle=linestyle, linewidth=linewidth, label=label))


def _overlay_anomaly(ax: Any, world: dict[str, Any], *, label: str = "true anomaly") -> None:
    plt = _plt()
    anomaly = world["medium"]["anomaly"]
    kind = anomaly_kind(world)
    if kind in {"circle", "circle-layered"}:
        ax.add_patch(plt.Circle((float(anomaly["center_x"]), float(anomaly["center_z"])), float(anomaly["radius"]), fill=False, linewidth=2.0, label=label))
    elif kind == "ellipse":
        _add_ellipse_patch(
            ax,
            center_x=float(anomaly["center_x"]),
            center_z=float(anomaly["center_z"]),
            radius_x=float(anomaly["radius_x"]),
            radius_z=float(anomaly["radius_z"]),
            angle_degrees=float(anomaly["angle_degrees"]),
            label=label,
        )
    elif kind == "ring":
        ax.add_patch(plt.Circle((float(anomaly["center_x"]), float(anomaly["center_z"])), float(anomaly["outer_radius"]), fill=False, linewidth=2.0, label=label))
        ax.add_patch(plt.Circle((float(anomaly["center_x"]), float(anomaly["center_z"])), float(anomaly["inner_radius"]), fill=False, linewidth=1.5, label=None))
    elif kind == "crack":
        _add_crack_patch(
            ax,
            center_x=float(anomaly["center_x"]),
            center_z=float(anomaly["center_z"]),
            length=float(anomaly["length"]),
            width=float(anomaly["width"]),
            angle_degrees=float(anomaly["angle_degrees"]),
            label=label,
        )
    elif kind == "rectangle":
        width = float(anomaly["width"])
        height = float(anomaly["height"])
        lower_left = (float(anomaly["center_x"]) - width / 2.0, float(anomaly["center_z"]) - height / 2.0)
        ax.add_patch(plt.Rectangle(lower_left, width, height, fill=False, linewidth=2.0, label=label))
    elif kind == "two-circles":
        for idx, circle in enumerate(anomaly["circles"]):
            ax.add_patch(plt.Circle((float(circle["center_x"]), float(circle["center_z"])), float(circle["radius"]), fill=False, linewidth=1.5, label=label if idx == 0 else None))
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
    blind_public = is_blind_public_world(world)
    velocity_model = background_velocity_model_from_world(world) if blind_public else velocity_model_from_world(world)
    plt = _plt()
    out = ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(velocity_model.T, origin="lower", extent=_domain_extent(world), aspect="equal")
    fig.colorbar(image, ax=ax, label="velocity")
    if not blind_public:
        _overlay_anomaly(ax, world)
    else:
        ax.text(0.02, 0.02, "blind public world: answer hidden", transform=ax.transAxes, fontsize=8, va="bottom")
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
    blind_public = bool(reconstruction.get("answer_hidden", False)) or is_blind_public_world(world)
    velocity_model = background_velocity_model_from_world(world) if blind_public else velocity_model_from_world(world)
    best = reconstruction.get("best_candidate", {})
    true = {} if blind_public else (reconstruction.get("true_center", {}) or {})
    objective = reconstruction.get("objective", {})
    plt = _plt()
    out = ensure_parent(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    ax0 = axes[0]
    image0 = ax0.imshow(velocity_model.T, origin="lower", extent=_domain_extent(world), aspect="equal")
    fig.colorbar(image0, ax=ax0, label="velocity")
    if not blind_public:
        _overlay_anomaly(ax0, world, label="true")
    else:
        ax0.text(0.02, 0.02, "true anomaly hidden", transform=ax0.transAxes, fontsize=8, va="bottom")
    if best:
        pred_kind = str(best.get("kind") or reconstruction.get("target_kind") or anomaly_kind(world))
        if pred_kind == "ellipse" and "radius_x" in best and "radius_z" in best:
            label = f"reconstructed ellipse"
            _add_ellipse_patch(
                ax0,
                center_x=float(best["center_x"]),
                center_z=float(best["center_z"]),
                radius_x=float(best["radius_x"]),
                radius_z=float(best["radius_z"]),
                angle_degrees=float(best.get("angle_degrees", 0.0)),
                label=label,
                linestyle="--",
                linewidth=2.0,
            )
        elif pred_kind in {"circle", "circle-layered"} and "radius" in best:
            label = f"reconstructed r={float(best.get('radius', 0.0)):.3f}"
            ax0.add_patch(plt.Circle((float(best["center_x"]), float(best["center_z"])), float(best["radius"]), fill=False, linestyle="--", linewidth=2.0, label=label))
        else:
            ax0.scatter([float(best["center_x"])], [float(best["center_z"])], marker="x", s=90, label="best center")
    ax0.set_title("Reconstructed anomaly" if blind_public else "True and reconstructed anomaly")
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
    """Return a center-probability grid using v0.5.1 unique-center deduplication."""
    candidates = list(reconstruction.get("candidates", []))
    if not candidates:
        return np.asarray([]), np.asarray([]), np.asarray([[]])
    try:
        from .uncertainty import candidate_probabilities

        summary = candidate_probabilities(reconstruction, temperature=temperature)
        centers = list(summary.get("center_probabilities", []))
    except Exception:
        centers = []
        best_by_center: dict[tuple[float, float], float] = {}
        mismatches: list[float] = []
        for c in candidates:
            try:
                key = (round(float(c["center_x"]), 8), round(float(c["center_z"]), 8))
                mismatch = float(c["mismatch"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(mismatch):
                continue
            mismatches.append(mismatch)
            best_by_center[key] = min(best_by_center.get(key, float("inf")), mismatch)
        if best_by_center:
            arr = np.asarray(list(best_by_center.values()), dtype=float)
            temp = float(temperature or max(float(np.nanpercentile(arr, 75.0) - float(np.nanmin(arr))), 1.0e-9))
            weights = np.exp(-(arr - float(np.nanmin(arr))) / temp)
            weights = weights / max(float(weights.sum()), 1.0e-12)
            centers = [
                {"center_x": key[0], "center_z": key[1], "probability": float(prob), "mismatch": float(mismatch)}
                for (key, mismatch), prob in zip(best_by_center.items(), weights)
            ]
    if not centers:
        return np.asarray([]), np.asarray([]), np.asarray([[]])
    xs = np.asarray(sorted({round(float(c["center_x"]), 8) for c in centers}), dtype=float)
    zs = np.asarray(sorted({round(float(c["center_z"]), 8) for c in centers}), dtype=float)
    grid = np.zeros((len(zs), len(xs)), dtype=float)
    x_index = {round(float(x), 8): i for i, x in enumerate(xs)}
    z_index = {round(float(z), 8): i for i, z in enumerate(zs)}
    for c in centers:
        try:
            ix = x_index[round(float(c["center_x"]), 8)]
            iz = z_index[round(float(c["center_z"]), 8)]
            grid[iz, ix] = float(c.get("probability", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
    return xs, zs, grid

def _uncertainty_summary_for_title(reconstruction: dict[str, Any], temperature: float | None) -> dict[str, Any]:
    summary = reconstruction.get("uncertainty") or {}
    if (
        temperature is None
        and isinstance(summary, dict)
        and summary.get("effective_candidates") is not None
        and summary.get("center_probability_mode") == "unique-center-min-mismatch"
    ):
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
    blind_public = bool(reconstruction.get("answer_hidden", False)) or is_blind_public_world(world)
    true = {} if blind_public else (reconstruction.get("true_center", {}) or {})
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
