"""Active sensing demo utilities for WaveSleuth-Devito.

v0.7 added a tiny multi-round loop:

1. start with one source and a fixed receiver ring
2. simulate observations with cumulative sources
3. invert the current observations
4. use the reconstruction/uncertainty to pick the next source
5. repeat and report whether the reconstruction improved

v0.7.2 keeps simulate_world backward-compatible while active artifacts
standardize saved sequential runs to consistent shot/time/receiver trace shapes, each round records uncertainty diagnostics, and
active runs can be compared with an active leaderboard.
"""

from __future__ import annotations

import copy
import html
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .exceptions import UnsupportedWorldError, ValidationError
from .geometry import grid_extent, receiver_coordinates
from .inversion import grid_search_circle, grid_search_ellipse
from .io import ensure_parent, load_json, save_json, save_world
from .metadata import base_metadata
from .scoring import score_reconstruction
from .simulation import simulate_world
from .uncertainty import candidate_probabilities
from .visualization import visualize_reconstruction, visualize_run, visualize_uncertainty, visualize_world
from .world import acquisition_preset, anomaly_kind, make_default_world, make_demo_world, validate_world

SUPPORTED_ACTIVE_KINDS = ("circle", "ellipse")
SUPPORTED_ACTIVE_STRATEGIES = ("uncertainty", "spread", "opposite-best")


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _copy_point(point: dict[str, Any]) -> dict[str, float]:
    return {"x": float(point["x"]), "z": float(point["z"])}


def _distance(a: dict[str, float] | tuple[float, float], b: dict[str, float] | tuple[float, float]) -> float:
    if isinstance(a, dict):
        ax, az = float(a["x"]), float(a["z"])
    else:
        ax, az = float(a[0]), float(a[1])
    if isinstance(b, dict):
        bx, bz = float(b["x"]), float(b["z"])
    else:
        bx, bz = float(b[0]), float(b[1])
    return math.hypot(ax - bx, az - bz)


def _same_point(a: dict[str, float], b: dict[str, float], *, tol: float = 1.0e-6) -> bool:
    return _distance(a, b) <= tol


def _unique_points(points: Iterable[dict[str, Any]], *, tol: float = 1.0e-6) -> list[dict[str, float]]:
    unique: list[dict[str, float]] = []
    for point in points:
        p = _copy_point(point)
        if not any(_same_point(p, q, tol=tol) for q in unique):
            unique.append(p)
    return unique


def default_active_world(kind: str = "circle") -> dict[str, Any]:
    """Return a small world configured for active-sensing demos."""
    kind = str(kind)
    if kind not in SUPPORTED_ACTIVE_KINDS:
        raise UnsupportedWorldError(f"Active demo supports {SUPPORTED_ACTIVE_KINDS}, got {kind!r}.")

    if kind == "circle":
        world = make_demo_world()
    else:
        world = make_default_world("ellipse", acquisition="ring", name="active_ellipse_demo")

    ring = acquisition_preset("ring")
    world["name"] = f"active_{kind}_demo"
    world["grid"].update({"nx": 52, "nz": 52, "extent_x": 1.0, "extent_z": 1.0})
    world["simulation"].update(
        {
            "nt": 340,
            "dt": 0.0015,
            "space_order": 4,
            "source_frequency": 18.0,
            "shot_mode": "sequential",
            "boundary": "sponge",
            "sponge_width": 5,
            "sponge_strength": 12.0,
        }
    )
    world["acquisition"] = {
        "sources": [{"x": 0.18, "z": 0.18}],
        "receivers": [_copy_point(p) for p in ring["receivers"]],
    }
    validate_world(world)
    return world


def active_source_pool(world: dict[str, Any], *, preset: str = "boundary-8") -> list[dict[str, float]]:
    """Return deterministic candidate source locations for active sensing."""
    validate_world(world)
    extent_x, extent_z = grid_extent(world)
    if preset not in {"boundary-8", "boundary-12"}:
        raise ValidationError("active source pool preset must be 'boundary-8' or 'boundary-12'.")

    ex = float(extent_x)
    ez = float(extent_z)
    base = [
        {"x": 0.18 * ex, "z": 0.18 * ez},
        {"x": 0.50 * ex, "z": 0.14 * ez},
        {"x": 0.82 * ex, "z": 0.18 * ez},
        {"x": 0.86 * ex, "z": 0.50 * ez},
        {"x": 0.82 * ex, "z": 0.82 * ez},
        {"x": 0.50 * ex, "z": 0.86 * ez},
        {"x": 0.18 * ex, "z": 0.82 * ez},
        {"x": 0.14 * ex, "z": 0.50 * ez},
    ]
    if preset == "boundary-12":
        base.extend(
            [
                {"x": 0.32 * ex, "z": 0.14 * ez},
                {"x": 0.68 * ex, "z": 0.14 * ez},
                {"x": 0.68 * ex, "z": 0.86 * ez},
                {"x": 0.32 * ex, "z": 0.86 * ez},
            ]
        )
    return _unique_points(base)


def _best_center(reconstruction: dict[str, Any]) -> tuple[float, float] | None:
    best = reconstruction.get("best_candidate", {})
    try:
        return float(best["center_x"]), float(best["center_z"])
    except (KeyError, TypeError, ValueError):
        return None


def uncertainty_metrics_from_reconstruction(reconstruction: dict[str, Any]) -> dict[str, Any]:
    """Return compact uncertainty diagnostics for active summaries.

    These values are display/decision aids, not calibrated Bayesian posterior
    quantities. Missing or malformed candidate lists return a small unsupported
    record instead of failing the active demo.
    """
    try:
        probabilities = candidate_probabilities(reconstruction)
    except Exception as exc:
        return {
            "supported": False,
            "message": f"uncertainty unavailable: {exc}",
        }

    center_probs = probabilities.get("center_probabilities", [])
    top_center = center_probs[0] if center_probs else {}
    metrics = {
        "supported": True,
        "temperature": float(probabilities.get("temperature", 0.0)),
        "n_candidates": int(probabilities.get("n_candidates", 0)),
        "n_centers": int(probabilities.get("n_centers", 0)),
        "duplicate_center_candidates": int(probabilities.get("duplicate_center_candidates", 0)),
        "effective_candidates": float(probabilities.get("effective_candidates", 0.0)),
        "center_effective_candidates": float(
            probabilities.get("center_effective_candidates", probabilities.get("center_entropy_effective_candidates", 0.0))
        ),
        "center_entropy_effective_candidates": float(probabilities.get("center_entropy_effective_candidates", 0.0)),
        "center_top_probability": float(probabilities.get("center_top_probability", 0.0)),
        "top_3_center_probability_mass": float(probabilities.get("top_3_center_probability_mass", 0.0)),
        "top_5_center_probability_mass": float(probabilities.get("top_5_center_probability_mass", 0.0)),
        "top_center": {
            "center_x": float(top_center.get("center_x", 0.0)) if top_center else None,
            "center_z": float(top_center.get("center_z", 0.0)) if top_center else None,
            "probability": float(top_center.get("probability", 0.0)) if top_center else None,
            "mismatch": float(top_center.get("mismatch", 0.0)) if top_center else None,
        },
    }
    return metrics


def estimated_center_from_reconstruction(reconstruction: dict[str, Any]) -> dict[str, Any]:
    """Estimate target center from uncertainty, falling back to the best candidate."""
    try:
        probabilities = candidate_probabilities(reconstruction)
        centers = probabilities.get("center_probabilities", [])
        if centers:
            total = float(sum(float(c.get("probability", 0.0)) for c in centers))
            if total > 0.0:
                x = sum(float(c["center_x"]) * float(c.get("probability", 0.0)) for c in centers) / total
                z = sum(float(c["center_z"]) * float(c.get("probability", 0.0)) for c in centers) / total
                top = centers[0]
                return {
                    "center_x": float(x),
                    "center_z": float(z),
                    "source": "uncertainty-weighted-center",
                    "top_center_x": float(top["center_x"]),
                    "top_center_z": float(top["center_z"]),
                    "top_center_probability": float(top.get("probability", 0.0)),
                    "center_effective_candidates": float(
                        probabilities.get("center_effective_candidates", probabilities.get("center_entropy_effective_candidates", 0.0))
                    ),
                }
    except Exception:
        pass

    best = _best_center(reconstruction)
    if best is not None:
        return {"center_x": best[0], "center_z": best[1], "source": "best-candidate"}
    return {"center_x": 0.5, "center_z": 0.5, "source": "domain-center-fallback"}


def select_next_source(
    world: dict[str, Any],
    reconstruction: dict[str, Any],
    used_sources: list[dict[str, Any]],
    *,
    strategy: str = "uncertainty",
    pool_preset: str = "boundary-8",
) -> dict[str, Any]:
    """Choose the next source location from a deterministic boundary pool.

    The strategy is intentionally simple. It favors unused sources that add
    geometric spread relative to existing sources. The uncertainty strategy also
    favors sources far from the current uncertainty-weighted target center so the
    next shot illuminates the candidate region from a new direction.
    """
    strategy = str(strategy)
    if strategy not in SUPPORTED_ACTIVE_STRATEGIES:
        raise ValidationError(f"Unsupported active strategy {strategy!r}.")
    pool = active_source_pool(world, preset=pool_preset)
    used = _unique_points(used_sources)
    choices = [p for p in pool if not any(_same_point(p, q) for q in used)]
    if not choices:
        raise ValidationError("No unused active source candidates remain.")

    estimate = estimated_center_from_reconstruction(reconstruction)
    target = (float(estimate["center_x"]), float(estimate["center_z"]))
    extent_x, extent_z = grid_extent(world)
    diag = max(math.hypot(float(extent_x), float(extent_z)), 1.0e-12)

    scored: list[dict[str, Any]] = []
    for point in choices:
        if used:
            min_existing = min(_distance(point, existing) for existing in used) / diag
            mean_existing = sum(_distance(point, existing) for existing in used) / (diag * len(used))
        else:
            min_existing = 1.0
            mean_existing = 1.0
        target_distance = _distance(point, target) / diag

        if strategy == "spread":
            score = 0.85 * min_existing + 0.15 * mean_existing
        elif strategy == "opposite-best":
            score = 0.65 * target_distance + 0.35 * min_existing
        else:
            uncertainty_bonus = min(1.0, float(estimate.get("center_effective_candidates", 1.0)) / 20.0)
            score = (0.45 + 0.20 * uncertainty_bonus) * min_existing + 0.35 * target_distance + 0.20 * mean_existing
        row = {
            "x": float(point["x"]),
            "z": float(point["z"]),
            "selection_score": float(score),
            "distance_to_estimated_center": float(target_distance),
            "min_distance_to_existing_sources": float(min_existing),
            "strategy": strategy,
            "estimated_center": estimate,
        }
        scored.append(row)
    scored.sort(key=lambda item: (-float(item["selection_score"]), float(item["x"]), float(item["z"])))
    winner = dict(scored[0])
    winner["ranked_candidates"] = scored
    return winner


def _round_world(secret_world: dict[str, Any], sources: list[dict[str, Any]], *, name: str) -> dict[str, Any]:
    world = copy.deepcopy(secret_world)
    world["name"] = name
    world["acquisition"]["sources"] = [_copy_point(p) for p in sources]
    world["simulation"]["shot_mode"] = "sequential"
    validate_world(world)
    return world


def _invert_round(
    world: dict[str, Any],
    run_path: Path,
    recon_path: Path,
    *,
    candidate_grid_size: int,
    refine_levels: int,
    quiet: bool,
) -> dict[str, Any]:
    kind = anomaly_kind(world)
    if kind == "circle":
        return grid_search_circle(
            run_path,
            out_path=recon_path,
            candidate_grid_size=candidate_grid_size,
            refine_levels=refine_levels,
            mismatch_mode="differential",
            metric="l2",
            shot_mode="sequential",
            quiet=quiet,
        )
    if kind == "ellipse":
        return grid_search_ellipse(
            run_path,
            out_path=recon_path,
            candidate_grid_size=candidate_grid_size,
            refine_levels=refine_levels,
            mismatch_mode="differential",
            metric="l2",
            shot_mode="sequential",
            quiet=quiet,
        )
    raise UnsupportedWorldError("Active demo currently supports circle and ellipse worlds.")


def _score_value(score: dict[str, Any]) -> float | None:
    try:
        return float(score.get("reconstruction_score", score.get("iou")))
    except (TypeError, ValueError):
        return None


def _compact_best(reconstruction: dict[str, Any]) -> dict[str, Any]:
    best = reconstruction.get("best_candidate", {})
    keys = ["kind", "center_x", "center_z", "radius", "radius_x", "radius_z", "angle_degrees", "anomaly_velocity", "mismatch"]
    out: dict[str, Any] = {}
    for key in keys:
        if key in best:
            value = best[key]
            out[key] = round(float(value), 6) if isinstance(value, (int, float)) else value
    return out



def _standardize_active_run_trace_shape(run_path: str | Path) -> bool:
    """Promote active single-shot trace files from 2D to 3D on disk.

    `simulate_world()` intentionally keeps its historical single-shot API:
    `(time, receiver)`. Active sensing needs a stable per-round file format so
    round 1 and later rounds are shaped consistently as `(shot, time, receiver)`.
    This helper performs that active-only normalization after the run is saved.

    Returns True when the file was rewritten, False when it already had a 3D
    active trace layout.
    """
    try:
        with np.load(run_path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
    except Exception as exc:
        raise ValidationError(f"Could not load active run file {run_path}: {exc}") from exc

    if "receiver_traces" not in arrays:
        raise ValidationError(f"Active run file {run_path} is missing receiver_traces.")

    traces = np.asarray(arrays["receiver_traces"])
    if traces.ndim == 3:
        return False
    if traces.ndim != 2:
        raise ValidationError(
            f"Active run file {run_path} has unsupported receiver_traces shape {traces.shape}; "
            "expected 2D or 3D."
        )

    arrays["receiver_traces"] = traces[np.newaxis, :, :]
    np.savez_compressed(run_path, **arrays)
    return True

def _run_trace_metadata(run_path: str | Path) -> dict[str, Any]:
    """Return small metadata about a run file's receiver trace array."""
    try:
        with np.load(run_path, allow_pickle=False) as data:
            traces = np.asarray(data["receiver_traces"])
    except Exception as exc:
        return {"available": False, "message": str(exc)}
    meta: dict[str, Any] = {
        "available": True,
        "shape": [int(v) for v in traces.shape],
        "ndim": int(traces.ndim),
    }
    if traces.ndim == 3:
        meta.update(
            {
                "layout": "shot_time_receiver",
                "n_shots": int(traces.shape[0]),
                "nt": int(traces.shape[1]),
                "n_receivers": int(traces.shape[2]),
                "active_trace_shape_standard": True,
            }
        )
    elif traces.ndim == 2:
        meta.update(
            {
                "layout": "time_receiver",
                "n_shots": 1,
                "nt": int(traces.shape[0]),
                "n_receivers": int(traces.shape[1]),
                "active_trace_shape_standard": False,
            }
        )
    else:
        meta.update({"layout": "unknown", "active_trace_shape_standard": False})
    return meta


def _write_progress_plot(summary: dict[str, Any], out_path: str | Path) -> Path:
    rounds = summary.get("rounds", [])
    out = ensure_parent(out_path)
    plt = _plt()
    xs = [int(r.get("round", i + 1)) for i, r in enumerate(rounds)]
    ious = [float(r.get("physical_score", {}).get("iou", 0.0)) for r in rounds]
    center_errors = [float(r.get("physical_score", {}).get("center_error", 0.0)) for r in rounds]

    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(xs, ious, marker="o", label="IoU")
    ax1.set_xlabel("active round")
    ax1.set_ylabel("IoU")
    ax1.set_ylim(0.0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(xs, center_errors, marker="s", linestyle="--", label="center error")
    ax2.set_ylabel("center error")
    ax1.set_title("Active sensing progress")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _write_source_layout_plot(secret_world: dict[str, Any], source_history: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Write a compact plot showing selected sources by active round."""
    out = ensure_parent(out_path)
    plt = _plt()
    extent_x, extent_z = grid_extent(secret_world)
    rec = receiver_coordinates(secret_world)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(rec[:, 0], rec[:, 1], marker="v", s=42, label="receivers")
    if source_history:
        xs = [float(p["x"]) for p in source_history]
        zs = [float(p["z"]) for p in source_history]
        ax.plot(xs, zs, marker="*", linewidth=1.2, label="selected sources")
        for idx, point in enumerate(source_history, start=1):
            ax.annotate(str(idx), (float(point["x"]), float(point["z"])), xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(0.0, float(extent_x))
    ax.set_ylim(0.0, float(extent_z))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Active source layout")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _write_active_report(summary: dict[str, Any], out_path: str | Path) -> Path:
    out = ensure_parent(out_path)
    root = out.parent

    def rel(path: str | None) -> str:
        if not path:
            return ""
        try:
            return html.escape(os.path.relpath(str(Path(path).resolve()), str(root.resolve())))
        except Exception:
            return html.escape(str(path))

    rows = []
    for item in summary.get("rounds", []):
        score = item.get("physical_score", {})
        unc = item.get("uncertainty_summary", {})
        trace_meta = item.get("trace_metadata", {})
        iou = score.get("iou", "")
        center_error = score.get("center_error", "")
        iou_text = f"{float(iou):.4f}" if isinstance(iou, (int, float)) else html.escape(str(iou))
        center_text = f"{float(center_error):.4f}" if isinstance(center_error, (int, float)) else html.escape(str(center_error))
        center_eff = unc.get("center_effective_candidates", "")
        center_eff_text = f"{float(center_eff):.2f}" if isinstance(center_eff, (int, float)) else html.escape(str(center_eff))
        top_prob = unc.get("center_top_probability", "")
        top_prob_text = f"{float(top_prob):.3f}" if isinstance(top_prob, (int, float)) else html.escape(str(top_prob))
        rows.append(
            "<tr>"
            f"<td>{item.get('round')}</td>"
            f"<td>{item.get('n_sources')}</td>"
            f"<td>{iou_text}</td>"
            f"<td>{center_text}</td>"
            f"<td>{center_eff_text}</td>"
            f"<td>{top_prob_text}</td>"
            f"<td>{html.escape(str(trace_meta.get('shape', '')))}</td>"
            f"<td>{html.escape(str(item.get('best_candidate', {})))}</td>"
            "</tr>"
        )

    image_blocks = []
    progress = rel(summary.get("figures", {}).get("progress"))
    if progress:
        image_blocks.append(f"<h2>Progress</h2><img src='{progress}' alt='progress' style='max-width: 760px;'>")
    layout = rel(summary.get("figures", {}).get("source_layout"))
    if layout:
        image_blocks.append(f"<h2>Selected source layout</h2><img src='{layout}' alt='source layout' style='max-width: 760px;'>")
    for item in summary.get("rounds", []):
        figs = item.get("figures", {})
        image_blocks.append(f"<h2>Round {item.get('round')}</h2>")
        for label, key in [("World", "world"), ("Traces", "traces"), ("Reconstruction", "reconstruction"), ("Uncertainty", "uncertainty")]:
            path = rel(figs.get(key))
            if path:
                image_blocks.append(f"<h3>{label}</h3><img src='{path}' alt='{label}' style='max-width: 760px;'>")

    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>WaveSleuth active sensing report</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; line-height: 1.45; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.55rem; vertical-align: top; }}
code, pre {{ background: #f4f4f4; padding: 0.1rem 0.25rem; }}
img {{ border: 1px solid #ddd; margin-bottom: 1rem; }}
</style>
</head>
<body>
<h1>WaveSleuth active sensing report</h1>
<p><strong>Version:</strong> {html.escape(str(summary.get('version', '')))}</p>
<p><strong>Kind:</strong> {html.escape(str(summary.get('kind', '')))}</p>
<p><strong>Strategy:</strong> {html.escape(str(summary.get('strategy', '')))}</p>
<p><strong>Final IoU:</strong> {summary.get('final_physical_score', {}).get('iou', '')}</p>
<p><strong>Final center error:</strong> {summary.get('final_physical_score', {}).get('center_error', '')}</p>
<p><strong>Score delta:</strong> {summary.get('score_delta', '')}</p>
<h2>Round summary</h2>
<table>
<tr><th>Round</th><th>Sources</th><th>IoU</th><th>Center error</th><th>Center eff.</th><th>Top center prob.</th><th>Trace shape</th><th>Best candidate</th></tr>
{''.join(rows)}
</table>
{''.join(image_blocks)}
<h2>Notes</h2>
<ul>
<li>This v0.7.2 active demo re-simulates cumulative shots each round for simplicity.</li>
<li>Active trace files are standardized to <code>(shot, time, receiver)</code> after simulation, including one-shot rounds.</li>
<li>The source-selection policy is deterministic and heuristic, not an optimal experimental-design solver.</li>
<li>Use this to inspect whether additional illumination reduces ambiguity and improves reconstruction.</li>
</ul>
</body>
</html>
"""
    out.write_text(html_text, encoding="utf-8")
    return out


def run_active_demo(
    out_dir: str | Path,
    *,
    kind: str = "circle",
    rounds: int = 3,
    candidate_grid_size: int = 5,
    refine_levels: int = 1,
    strategy: str = "uncertainty",
    pool_preset: str = "boundary-8",
    noise_level: float = 0.0,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run a small multi-round active-sensing demo."""
    if int(rounds) < 1:
        raise ValidationError("active-demo rounds must be at least 1.")
    if int(candidate_grid_size) < 2:
        raise ValidationError("candidate_grid_size must be at least 2.")
    if int(refine_levels) < 0:
        raise ValidationError("refine_levels must be non-negative.")
    if float(noise_level) < 0.0:
        raise ValidationError("noise_level must be non-negative.")
    if strategy not in SUPPORTED_ACTIVE_STRATEGIES:
        raise ValidationError(f"Unsupported active strategy {strategy!r}.")

    root = Path(out_dir)
    worlds_dir = root / "worlds"
    runs_dir = root / "runs"
    rounds_dir = root / "rounds"
    figures_dir = root / "figures"
    reports_dir = root / "reports"
    for directory in (worlds_dir, runs_dir, rounds_dir, figures_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    secret_world = default_active_world(kind)
    answer_world_path = worlds_dir / "active_secret_world.json"
    save_world(secret_world, answer_world_path)
    visualize_world(secret_world, figures_dir / "answer_world.png")

    source_history = [_copy_point(secret_world["acquisition"]["sources"][0])]
    round_records: list[dict[str, Any]] = []
    t0 = perf_counter()

    for index in range(int(rounds)):
        round_number = index + 1
        stem = f"round_{round_number:02d}"
        current_world = _round_world(secret_world, source_history, name=f"active_{kind}_{stem}")
        round_dir = rounds_dir / stem
        round_dir.mkdir(parents=True, exist_ok=True)
        world_path = round_dir / "world.json"
        run_path = runs_dir / f"{stem}_obs.npz"
        recon_path = runs_dir / f"{stem}_recon.json"
        save_world(current_world, world_path)

        if not quiet:
            print(f"active round {round_number}/{rounds}: sources={len(source_history)}")

        simulate_world(
            current_world,
            out_path=str(run_path),
            save_wavefield=False,
            quiet=quiet,
            shot_mode="sequential",
            noise_level=float(noise_level) if noise_level > 0.0 else None,
            noise_seed=7000 + round_number,
        )
        reconstruction = _invert_round(
            current_world,
            run_path,
            recon_path,
            candidate_grid_size=int(candidate_grid_size),
            refine_levels=int(refine_levels),
            quiet=quiet,
        )
        _standardize_active_run_trace_shape(run_path)
        trace_metadata = _run_trace_metadata(run_path)
        physical_score = score_reconstruction(secret_world, reconstruction)
        uncertainty_summary = uncertainty_metrics_from_reconstruction(reconstruction)
        reconstruction["physical_score"] = physical_score
        reconstruction["uncertainty_summary"] = uncertainty_summary
        reconstruction.setdefault("active", {})["trace_metadata"] = trace_metadata
        reconstruction.setdefault("active", {})["round"] = int(round_number)
        save_json(reconstruction, recon_path)

        figs = {
            "world": str(figures_dir / f"{stem}_world.png"),
            "traces": str(figures_dir / f"{stem}_traces.png"),
            "reconstruction": str(figures_dir / f"{stem}_reconstruction.png"),
            "uncertainty": str(figures_dir / f"{stem}_uncertainty.png"),
        }
        visualize_world(current_world, figs["world"])
        visualize_run(run_path, figs["traces"])
        visualize_reconstruction(reconstruction, figs["reconstruction"])
        try:
            visualize_uncertainty(reconstruction, figs["uncertainty"])
        except Exception as exc:
            if not quiet:
                print(f"warning: could not visualize uncertainty for {stem}: {exc}")
            figs.pop("uncertainty", None)

        selection: dict[str, Any] | None = None
        if round_number < int(rounds):
            selection = select_next_source(
                current_world,
                reconstruction,
                source_history,
                strategy=strategy,
                pool_preset=pool_preset,
            )
            source_history.append({"x": float(selection["x"]), "z": float(selection["z"])})

        record = {
            "round": int(round_number),
            "n_sources": int(len(current_world["acquisition"]["sources"])),
            "sources": [_copy_point(p) for p in current_world["acquisition"]["sources"]],
            "world_path": str(world_path),
            "run_path": str(run_path),
            "reconstruction_path": str(recon_path),
            "figures": figs,
            "trace_metadata": trace_metadata,
            "best_candidate": _compact_best(reconstruction),
            "physical_score": physical_score,
            "uncertainty_summary": uncertainty_summary,
            "selected_next_source": selection,
        }
        round_records.append(record)

    elapsed = perf_counter() - t0
    first_score = _score_value(round_records[0].get("physical_score", {})) if round_records else None
    final_score = _score_value(round_records[-1].get("physical_score", {})) if round_records else None
    final_physical = round_records[-1].get("physical_score", {}) if round_records else {}
    progress_path = _write_progress_plot({"rounds": round_records}, figures_dir / "active_progress.png")
    source_layout_path = _write_source_layout_plot(secret_world, source_history, figures_dir / "active_source_layout.png")

    summary: dict[str, Any] = {
        **base_metadata(),
        "version": "0.7.2",
        "mode": "active-demo",
        "kind": kind,
        "strategy": strategy,
        "pool_preset": pool_preset,
        "round_count": int(rounds),
        "candidate_grid_size": int(candidate_grid_size),
        "refine_levels": int(refine_levels),
        "noise_level": float(noise_level),
        "secret_world_path": str(answer_world_path),
        "source_history": source_history,
        "rounds": round_records,
        "runtime_seconds": float(elapsed),
        "initial_reconstruction_score": first_score,
        "final_reconstruction_score": final_score,
        "score_delta": None if first_score is None or final_score is None else float(final_score - first_score),
        "final_physical_score": final_physical,
        "final_uncertainty_summary": round_records[-1].get("uncertainty_summary", {}) if round_records else {},
        "figures": {
            "answer_world": str(figures_dir / "answer_world.png"),
            "progress": str(progress_path),
            "source_layout": str(source_layout_path),
        },
        "trace_layout": "shot_time_receiver",
        "notes": [
            "v0.7.2 active-demo is a deterministic heuristic active-sensing loop.",
            "It re-simulates cumulative sequential shots each round instead of storing incremental shot updates.",
            "Sequential active run files use a stable (shot, time, receiver) trace layout, including one-shot rounds.",
            "The selected next source is based on geometric spread and reconstruction uncertainty, not optimal experimental design.",
        ],
    }
    summary_path = root / "active_summary.json"
    save_json(summary, summary_path)
    report_path = _write_active_report(summary, reports_dir / "active_report.html")
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    save_json(summary, summary_path)
    return summary


def _active_summary_path(path: str | Path) -> Path:
    p = Path(path)
    return p / "active_summary.json" if p.is_dir() else p


def _round_float(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def active_score_from_summary(summary: dict[str, Any]) -> float:
    """Return a lightweight active-run score for comparing active demos.

    This is intentionally separate from challenge scoring. It rewards final IoU,
    penalizes normalized center error, and lightly penalizes source count.
    """
    final = summary.get("final_physical_score", {}) or {}
    iou = float(final.get("iou", final.get("reconstruction_score", 0.0)) or 0.0)
    norm_center = float(final.get("normalized_center_error", 0.0) or 0.0)
    n_sources = int(len(summary.get("source_history", [])) or summary.get("round_count", 0) or 0)
    return float(100.0 * iou - 20.0 * norm_center - 0.25 * n_sources)


def collect_active_leaderboard(paths: list[str | Path]) -> dict[str, Any]:
    """Collect and rank active-demo summaries."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        summary_path = _active_summary_path(path)
        try:
            summary = load_json(summary_path)
        except Exception as exc:
            rows.append({"path": str(summary_path), "error": str(exc)})
            continue
        final = summary.get("final_physical_score", {}) or {}
        initial = None
        rounds = summary.get("rounds", [])
        if rounds:
            initial = rounds[0].get("physical_score", {}) or {}
        unc = summary.get("final_uncertainty_summary", {}) or {}
        row = {
            "path": str(summary_path),
            "kind": summary.get("kind"),
            "strategy": summary.get("strategy"),
            "pool_preset": summary.get("pool_preset"),
            "round_count": int(summary.get("round_count", len(rounds) or 0)),
            "n_sources": int(len(summary.get("source_history", []))),
            "final_iou": _round_float(final.get("iou"), 4),
            "initial_iou": _round_float(initial.get("iou") if isinstance(initial, dict) else None, 4),
            "score_delta": _round_float(summary.get("score_delta"), 4),
            "center_error": _round_float(final.get("center_error"), 4),
            "normalized_center_error": _round_float(final.get("normalized_center_error"), 4),
            "center_effective_candidates": _round_float(unc.get("center_effective_candidates"), 3),
            "center_top_probability": _round_float(unc.get("center_top_probability"), 4),
            "runtime_seconds": _round_float(summary.get("runtime_seconds"), 3),
            "score": round(active_score_from_summary(summary), 3),
            "source_history": summary.get("source_history", []),
        }
        rows.append(row)
    rows.sort(
        key=lambda r: (
            1 if "error" in r else 0,
            -float(r.get("score", -1.0e99) or -1.0e99),
            -float(r.get("final_iou", -1.0) or -1.0),
            float(r.get("center_error", 1.0e99) or 1.0e99),
            str(r.get("strategy", "")),
        )
    )
    return {"active_leaderboard": rows}
