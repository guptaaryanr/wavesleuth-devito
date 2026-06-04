"""Experiment, challenge, comparison, and report helpers for v0.3."""

from __future__ import annotations

import copy
import html
import json
from pathlib import Path
from typing import Any

from .inversion import grid_search_circle
from .io import ensure_parent, load_json, load_world, save_json, save_world
from .scoring import budgeted_challenge_score, score_reconstruction
from .simulation import simulate_world
from .visualization import visualize_reconstruction, visualize_run, visualize_uncertainty, visualize_world
from .world import SUPPORTED_ACQUISITION_PRESETS, make_default_world, make_demo_world


def _n_forward_runs(reconstruction: dict[str, Any]) -> int:
    return int(reconstruction.get("candidate_grid", {}).get("evaluated_candidates", len(reconstruction.get("candidates", []))))


def run_challenge(
    name: str,
    out_dir: str | Path,
    *,
    candidate_grid_size: int = 5,
    refine_levels: int = 1,
    noise_level: float | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run a deterministic mini challenge and return a summary."""
    challenge = str(name).strip().lower()
    if challenge not in {"circle-easy", "circle-noisy", "circle-budget"}:
        raise ValueError("Supported challenges: circle-easy, circle-noisy, circle-budget")

    root = Path(out_dir)
    worlds = root / "worlds"
    runs = root / "runs"
    figures = root / "figures"
    reports = root / "reports"
    for directory in (worlds, runs, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)

    world = make_demo_world()
    world["name"] = f"challenge_{challenge}"
    used_noise = 0.0
    if challenge == "circle-noisy":
        used_noise = 0.035 if noise_level is None else float(noise_level)
    elif challenge == "circle-budget":
        used_noise = 0.015 if noise_level is None else float(noise_level)
        world["acquisition"] = make_default_world("circle", acquisition="left-right")["acquisition"]
        world["simulation"]["shot_mode"] = "sequential"
    elif noise_level is not None:
        used_noise = float(noise_level)

    world_path = worlds / f"{challenge}.json"
    run_path = runs / f"{challenge}_obs.npz"
    recon_path = runs / f"{challenge}_recon.json"
    report_path = reports / f"{challenge}_report.html"

    save_world(world, world_path)
    simulate_world(world, out_path=str(run_path), save_wavefield=True, quiet=quiet, noise_level=used_noise, noise_seed=404)
    reconstruction = grid_search_circle(
        run_path,
        out_path=recon_path,
        candidate_grid_size=candidate_grid_size,
        refine_levels=refine_levels,
        mismatch_mode="differential",
        quiet=quiet,
    )
    world_png = visualize_world(world, figures / f"{challenge}_world.png")
    traces_png = visualize_run(run_path, figures / f"{challenge}_traces.png")
    recon_png = visualize_reconstruction(reconstruction, figures / f"{challenge}_recon.png")
    uncertainty_png = visualize_uncertainty(reconstruction, figures / f"{challenge}_uncertainty.png")

    score = score_reconstruction(world, reconstruction)
    challenge_score = budgeted_challenge_score(
        score,
        n_forward_runs=_n_forward_runs(reconstruction),
        n_sources=len(world["acquisition"]["sources"]),
        n_receivers=len(world["acquisition"]["receivers"]),
    )
    summary = {
        "challenge": challenge,
        "world_path": str(world_path),
        "run_path": str(run_path),
        "reconstruction_path": str(recon_path),
        "figures": {
            "world": str(world_png),
            "traces": str(traces_png),
            "reconstruction": str(recon_png),
            "uncertainty": str(uncertainty_png),
        },
        "score": score,
        "challenge_score": challenge_score,
    }
    save_json(summary, runs / f"{challenge}_summary.json")
    write_reconstruction_report(reconstruction, report_path, extra_summary=summary)
    summary["report_path"] = str(report_path)
    return summary


def compare_acquisitions(
    world_or_path: dict[str, Any] | str | Path,
    *,
    out_dir: str | Path,
    presets: list[str],
    candidate_grid_size: int = 5,
    refine_levels: int = 0,
    mismatch_mode: str = "differential",
    metric: str = "l2",
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the same hidden circle world with several acquisition presets."""
    base_world = load_world(world_or_path) if isinstance(world_or_path, (str, Path)) else copy.deepcopy(world_or_path)
    root = Path(out_dir)
    worlds = root / "worlds"
    runs = root / "runs"
    figures = root / "figures"
    for directory in (worlds, runs, figures):
        directory.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for preset in presets:
        if preset not in SUPPORTED_ACQUISITION_PRESETS:
            raise ValueError(f"Unknown acquisition preset {preset!r}.")
        world = copy.deepcopy(base_world)
        world["name"] = f"{base_world.get('name', 'world')}_{preset}"
        world["acquisition"] = make_default_world("circle", acquisition=preset)["acquisition"]
        world.setdefault("simulation", {})["shot_mode"] = "simultaneous" if preset == "single" else "sequential"
        world_path = worlds / f"{preset}.json"
        run_path = runs / f"{preset}_obs.npz"
        recon_path = runs / f"{preset}_recon.json"
        save_world(world, world_path)
        simulate_world(world, out_path=str(run_path), save_wavefield=False, quiet=quiet)
        reconstruction = grid_search_circle(
            run_path,
            out_path=recon_path,
            candidate_grid_size=candidate_grid_size,
            refine_levels=refine_levels,
            mismatch_mode=mismatch_mode,
            metric=metric,
            quiet=quiet,
        )
        visualize_world(world, figures / f"{preset}_world.png")
        visualize_reconstruction(reconstruction, figures / f"{preset}_recon.png")
        score = score_reconstruction(world, reconstruction)
        n_forward = int(reconstruction.get("candidate_grid", {}).get("forward_runs", len(reconstruction.get("candidates", []))))
        row = {
            "preset": preset,
            "world_path": str(world_path),
            "run_path": str(run_path),
            "reconstruction_path": str(recon_path),
            "n_sources": len(world["acquisition"]["sources"]),
            "n_receivers": len(world["acquisition"]["receivers"]),
            "forward_runs": n_forward,
            "score": score,
            "best_candidate": reconstruction.get("best_candidate"),
            "uncertainty": reconstruction.get("uncertainty"),
            "challenge_score": budgeted_challenge_score(
                score,
                n_forward_runs=n_forward,
                n_sources=len(world["acquisition"]["sources"]),
                n_receivers=len(world["acquisition"]["receivers"]),
            ),
        }
        results.append(row)

    summary = {
        "kind": base_world.get("medium", {}).get("anomaly", {}).get("kind", "unknown"),
        "base_world": base_world.get("name", "unknown"),
        "presets": presets,
        "candidate_grid_size": int(candidate_grid_size),
        "refine_levels": int(refine_levels),
        "mismatch_mode": mismatch_mode,
        "metric": metric,
        "results": results,
    }
    save_json(summary, runs / "acquisition_comparison.json")
    return summary


def _html_table(mapping: dict[str, Any]) -> str:
    rows = []
    for key, value in mapping.items():
        if isinstance(value, (dict, list)):
            text = json.dumps(value, indent=2, sort_keys=True)
        else:
            text = str(value)
        rows.append(f"<tr><th>{html.escape(str(key))}</th><td><pre>{html.escape(text)}</pre></td></tr>")
    return "<table>" + "\n".join(rows) + "</table>"


def write_reconstruction_report(
    reconstruction_or_path: dict[str, Any] | str | Path,
    out_path: str | Path,
    *,
    extra_summary: dict[str, Any] | None = None,
) -> Path:
    """Write a small standalone HTML reconstruction report."""
    reconstruction = load_json(reconstruction_or_path) if isinstance(reconstruction_or_path, (str, Path)) else reconstruction_or_path
    best = reconstruction.get("best_candidate", {})
    score = reconstruction.get("score", {})
    uncertainty = reconstruction.get("uncertainty", {})
    objective = reconstruction.get("objective", {})
    search_params = reconstruction.get("search_parameters", {})
    out = ensure_parent(out_path)
    title = f"WaveSleuth report: {html.escape(str(reconstruction.get('world_name', 'unknown')))}"
    summary = {
        "method": reconstruction.get("method"),
        "objective": objective,
        "search_parameters": search_params,
        "best_candidate": best,
        "score": score,
        "uncertainty": uncertainty,
        "candidate_grid": reconstruction.get("candidate_grid", {}),
    }
    if extra_summary is not None:
        summary["extra_summary"] = extra_summary
    body = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; line-height: 1.45; margin: 2rem; max-width: 1100px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; vertical-align: top; padding: 0.5rem; }}
th {{ width: 220px; text-align: left; background: #f6f6f6; }}
pre {{ white-space: pre-wrap; margin: 0; }}
code {{ background: #f6f6f6; padding: 0.1rem 0.25rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>This is a lightweight v0.3 report. It records what was searched, what won, how it scored, and how concentrated the candidate mismatch surface was.</p>
{_html_table(summary)}
<h2>Notes</h2>
<ul>
<li>The uncertainty map is a soft diagnostic derived from mismatch values, not a formal posterior.</li>
<li>The boundary damping is a simple sponge layer, not a tuned PML.</li>
<li>Circle scoring focuses on center, radius, and mask IoU.</li>
</ul>
</body>
</html>
"""
    out.write_text(body, encoding="utf-8")
    return out
