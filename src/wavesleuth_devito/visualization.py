"""Matplotlib visualizations for worlds, runs, and reconstructions."""

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
        patch = plt.Circle(
            (float(anomaly["center_x"]), float(anomaly["center_z"])),
            float(anomaly["radius"]),
            fill=False,
            linewidth=2.0,
            label=label,
        )
        ax.add_patch(patch)
    elif kind == "rectangle":
        width = float(anomaly["width"])
        height = float(anomaly["height"])
        lower_left = (float(anomaly["center_x"]) - width / 2.0, float(anomaly["center_z"]) - height / 2.0)
        patch = plt.Rectangle(lower_left, width, height, fill=False, linewidth=2.0, label=label)
        ax.add_patch(patch)
    elif kind == "blobs":
        for idx, blob in enumerate(anomaly["blobs"]):
            patch = plt.Circle(
                (float(blob["center_x"]), float(blob["center_z"])),
                float(blob["radius"]),
                fill=False,
                linewidth=1.5,
                label=label if idx == 0 else None,
            )
            ax.add_patch(patch)
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
    image = ax.imshow(
        velocity_model.T,
        origin="lower",
        extent=_domain_extent(world),
        aspect="equal",
    )
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


def visualize_run(run_path: str | Path, out_path: str | Path) -> Path:
    """Plot receiver traces from a `.npz` run."""
    run = load_run_npz(run_path)
    world = world_from_run(run)
    traces = np.asarray(run["receiver_traces"], dtype=np.float32)
    time = np.asarray(run["time"], dtype=np.float32)
    plt = _plt()
    out = ensure_parent(out_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if traces.size == 0:
        ax.text(0.5, 0.5, "No traces found", ha="center", va="center")
    else:
        vmax = float(np.percentile(np.abs(traces), 99.0)) if np.any(traces) else 1.0
        image = ax.imshow(
            traces.T,
            origin="lower",
            aspect="auto",
            extent=[float(time[0]), float(time[-1]), -0.5, traces.shape[1] - 0.5],
            vmin=-vmax,
            vmax=vmax,
        )
        fig.colorbar(image, ax=ax, label="pressure amplitude")
    ax.set_xlabel("time")
    ax.set_ylabel("receiver index")
    ax.set_title(f"Receiver traces: {world.get('name', Path(run_path).stem)}")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def visualize_reconstruction(reconstruction_or_path: dict[str, Any] | str | Path, out_path: str | Path) -> Path:
    """Plot true/predicted anomalies and the candidate mismatch map."""
    reconstruction = load_json(reconstruction_or_path) if isinstance(reconstruction_or_path, (str, Path)) else reconstruction_or_path
    world = reconstruction.get("world")
    if not isinstance(world, dict):
        raise ValueError("Reconstruction does not contain embedded world metadata.")
    velocity_model = velocity_model_from_world(world)
    best = reconstruction.get("best_candidate", {})
    plt = _plt()
    out = ensure_parent(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    ax0 = axes[0]
    image0 = ax0.imshow(velocity_model.T, origin="lower", extent=_domain_extent(world), aspect="equal")
    fig.colorbar(image0, ax=ax0, label="velocity")
    _overlay_anomaly(ax0, world, label="true")
    if best:
        patch = plt.Circle(
            (float(best["center_x"]), float(best["center_z"])),
            float(best["radius"]),
            fill=False,
            linestyle="--",
            linewidth=2.0,
            label="reconstructed",
        )
        ax0.add_patch(patch)
    ax0.set_title("True and reconstructed anomaly")
    ax0.set_xlabel("x")
    ax0.set_ylabel("z")
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    mismatch_raw = reconstruction.get("mismatch_map", [])
    if mismatch_raw:
        mismatch = np.asarray(
            [[np.nan if value is None else float(value) for value in row] for row in mismatch_raw],
            dtype=np.float64,
        )
        grid = reconstruction.get("candidate_grid", {})
        xs = np.asarray(grid.get("xs", []), dtype=float)
        zs = np.asarray(grid.get("zs", []), dtype=float)
        if xs.size >= 2 and zs.size >= 2:
            extent = [float(xs.min()), float(xs.max()), float(zs.min()), float(zs.max())]
        else:
            extent = _domain_extent(world)
        image1 = ax1.imshow(mismatch, origin="lower", aspect="auto", extent=extent)
        fig.colorbar(image1, ax=ax1, label="normalized mismatch")
        if best:
            ax1.scatter([float(best["center_x"])], [float(best["center_z"])], marker="x", s=80)
    else:
        ax1.text(0.5, 0.5, "No mismatch map", ha="center", va="center")
    ax1.set_title("Candidate mismatch map")
    ax1.set_xlabel("candidate center x")
    ax1.set_ylabel("candidate center z")

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
