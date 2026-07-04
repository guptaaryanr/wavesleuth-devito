"""Active sensing demo utilities for WaveSleuth-Devito.

v0.7 adds a tiny multi-round loop:

1. start with one source and a fixed receiver ring
2. simulate observations with cumulative sources
3. invert the current observations
4. use the reconstruction/uncertainty to pick the next source
5. repeat and report whether the reconstruction improved

The implementation deliberately re-simulates cumulative shots each round. That is
simple, deterministic, and easy to inspect. More efficient incremental shot
reuse can come later.
"""

from __future__ import annotations

import copy
import html
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from .exceptions import UnsupportedWorldError, ValidationError
from .geometry import grid_extent
from .inversion import grid_search_circle, grid_search_ellipse
from .io import ensure_parent, save_json, save_world
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
        iou = score.get("iou", "")
        center_error = score.get("center_error", "")
        iou_text = f"{float(iou):.4f}" if isinstance(iou, (int, float)) else html.escape(str(iou))
        center_text = f"{float(center_error):.4f}" if isinstance(center_error, (int, float)) else html.escape(str(center_error))
        rows.append(
            "<tr>"
            f"<td>{item.get('round')}</td>"
            f"<td>{item.get('n_sources')}</td>"
            f"<td>{iou_text}</td>"
            f"<td>{center_text}</td>"
            f"<td>{html.escape(str(item.get('best_candidate', {})))}</td>"
            "</tr>"
        )

    image_blocks = []
    progress = rel(summary.get("figures", {}).get("progress"))
    if progress:
        image_blocks.append(f"<h2>Progress</h2><img src='{progress}' alt='progress' style='max-width: 760px;'>")
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
<h2>Round summary</h2>
<table>
<tr><th>Round</th><th>Sources</th><th>IoU</th><th>Center error</th><th>Best candidate</th></tr>
{''.join(rows)}
</table>
{''.join(image_blocks)}
<h2>Notes</h2>
<ul>
<li>This v0.7 active demo re-simulates cumulative shots each round for simplicity.</li>
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
        physical_score = score_reconstruction(secret_world, reconstruction)
        reconstruction["physical_score"] = physical_score
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
            "best_candidate": _compact_best(reconstruction),
            "physical_score": physical_score,
            "selected_next_source": selection,
        }
        round_records.append(record)

    elapsed = perf_counter() - t0
    first_score = _score_value(round_records[0].get("physical_score", {})) if round_records else None
    final_score = _score_value(round_records[-1].get("physical_score", {})) if round_records else None
    final_physical = round_records[-1].get("physical_score", {}) if round_records else {}
    progress_path = _write_progress_plot({"rounds": round_records}, figures_dir / "active_progress.png")

    summary: dict[str, Any] = {
        **base_metadata(),
        "version": "0.7.0",
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
        "figures": {"answer_world": str(figures_dir / "answer_world.png"), "progress": str(progress_path)},
        "notes": [
            "v0.7 active-demo is a deterministic heuristic active-sensing loop.",
            "It re-simulates cumulative sequential shots each round instead of storing incremental shot updates.",
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
