"""Small programmatic examples used by the CLI demo and curious users."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

from .inversion import grid_search_circle
from .io import load_world, save_json, save_world
from .report import generate_html_report
from .scoring import score_reconstruction
from .simulation import simulate_world
from .visualization import visualize_reconstruction, visualize_run, visualize_uncertainty, visualize_world
from .world import acquisition_preset, make_demo_world, validate_world


def run_demo(
    out_dir: str | Path,
    *,
    candidate_grid_size: int = 5,
    quiet: bool = False,
    refine_levels: int = 1,
    mismatch_mode: str = "differential",
    metric: str = "l2",
    search_strategy: str = "auto",
    top_k_refine: int = 5,
    final_refine_top_k: int = 1,
    search_radius: bool = False,
    search_velocity: bool = False,
    noise_level: float = 0.0,
) -> dict[str, Any]:
    """Run the tiny end-to-end WaveSleuth pipeline."""
    root = Path(out_dir)
    worlds = root / "worlds"
    runs = root / "runs"
    figures = root / "figures"
    reports = root / "reports"
    worlds.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    world = make_demo_world()
    world_path = worlds / "circle_demo.json"
    run_path = runs / "circle_obs.npz"
    recon_path = runs / "circle_recon.json"

    save_world(world, world_path)
    simulate_world(
        world,
        out_path=str(run_path),
        save_wavefield=True,
        quiet=quiet,
        shot_mode="sequential",
        noise_level=noise_level,
    )
    reconstruction = grid_search_circle(
        run_path,
        out_path=recon_path,
        candidate_grid_size=candidate_grid_size,
        refine_levels=refine_levels,
        mismatch_mode=mismatch_mode,
        metric=metric,
        search_strategy=search_strategy,
        top_k_refine=top_k_refine,
        final_refine_top_k=final_refine_top_k,
        search_radius=search_radius,
        search_velocity=search_velocity,
        quiet=quiet,
    )
    visualize_world(world, figures / "circle_world.png")
    visualize_run(run_path, figures / "circle_traces.png")
    visualize_reconstruction(reconstruction, figures / "circle_recon.png")
    visualize_uncertainty(reconstruction, figures / "circle_uncertainty.png")
    report_path = generate_html_report(recon_path, reports / "circle_report.html")
    final_score = score_reconstruction(world, reconstruction)
    return {
        "world_path": str(world_path),
        "run_path": str(run_path),
        "reconstruction_path": str(recon_path),
        "figures_dir": str(figures),
        "report_path": str(report_path),
        "score": final_score,
        "objective": reconstruction.get("objective", {}),
        "search": reconstruction.get("search", {}),
        "best_candidate": reconstruction.get("best_candidate", {}),
        "nearest_true_candidate": reconstruction.get("nearest_true_candidate", {}),
    }


def compare_acquisitions(
    world_path: str | Path,
    *,
    out_dir: str | Path,
    presets: Iterable[str],
    candidate_grid_size: int = 5,
    refine_levels: int = 0,
    mismatch_mode: str = "differential",
    metric: str = "l2",
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the same hidden world under several acquisition presets."""
    preset_list = list(presets)
    base_world = load_world(world_path)
    root = Path(out_dir)
    worlds = root / "worlds"
    runs = root / "runs"
    figures = root / "figures"
    worlds.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for preset in preset_list:
        world = copy.deepcopy(base_world)
        world["name"] = f"{base_world.get('name', 'world')}_{preset}"
        world["acquisition"] = acquisition_preset(preset)
        world.setdefault("simulation", {})["shot_mode"] = "simultaneous" if preset == "single" else "sequential"
        validate_world(world)
        safe = preset.replace("-", "_")
        wpath = worlds / f"{safe}.json"
        rpath = runs / f"{safe}_obs.npz"
        recon_path = runs / f"{safe}_recon.json"
        save_world(world, wpath)
        simulate_world(world, out_path=str(rpath), save_wavefield=False, quiet=quiet, shot_mode=world["simulation"]["shot_mode"])
        reconstruction = grid_search_circle(
            rpath,
            out_path=recon_path,
            candidate_grid_size=candidate_grid_size,
            refine_levels=refine_levels,
            mismatch_mode=mismatch_mode,
            metric=metric,
            quiet=quiet,
        )
        visualize_reconstruction(reconstruction, figures / f"{safe}_recon.png")
        score = score_reconstruction(world, reconstruction)
        results.append(
            {
                "preset": preset,
                "world_path": str(wpath),
                "run_path": str(rpath),
                "reconstruction_path": str(recon_path),
                "figure_path": str(figures / f"{safe}_recon.png"),
                "score": score,
                "best_candidate": reconstruction.get("best_candidate", {}),
                "objective": reconstruction.get("objective", {}),
                "forward_runs": reconstruction.get("candidate_grid", {}).get("forward_runs"),
            }
        )

    summary = {
        "world_path": str(world_path),
        "out_dir": str(root),
        "presets": preset_list,
        "results": results,
    }
    save_json(summary, root / "acquisition_comparison.json")
    return summary
