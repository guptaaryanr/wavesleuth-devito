"""Small programmatic examples used by the CLI demo and curious users."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .inversion import grid_search_circle
from .io import save_world
from .scoring import score_reconstruction
from .simulation import simulate_world
from .visualization import visualize_reconstruction, visualize_run, visualize_world
from .world import make_demo_world


def run_demo(
    out_dir: str | Path,
    *,
    candidate_grid_size: int = 5,
    quiet: bool = False,
    refine_levels: int = 1,
    mismatch_mode: str = "differential",
    metric: str = "l2",
) -> dict[str, Any]:
    """Run the tiny end-to-end WaveSleuth pipeline."""
    root = Path(out_dir)
    worlds = root / "worlds"
    runs = root / "runs"
    figures = root / "figures"
    worlds.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    world = make_demo_world()
    world_path = worlds / "circle_demo.json"
    run_path = runs / "circle_obs.npz"
    recon_path = runs / "circle_recon.json"

    save_world(world, world_path)
    simulate_world(world, out_path=str(run_path), save_wavefield=True, quiet=quiet, shot_mode="sequential")
    reconstruction = grid_search_circle(
        run_path,
        out_path=recon_path,
        candidate_grid_size=candidate_grid_size,
        refine_levels=refine_levels,
        mismatch_mode=mismatch_mode,
        metric=metric,
        quiet=quiet,
    )
    visualize_world(world, figures / "circle_world.png")
    visualize_run(run_path, figures / "circle_traces.png")
    visualize_reconstruction(reconstruction, figures / "circle_recon.png")
    final_score = score_reconstruction(world, reconstruction)
    return {
        "world_path": str(world_path),
        "run_path": str(run_path),
        "reconstruction_path": str(recon_path),
        "figures_dir": str(figures),
        "score": final_score,
        "objective": reconstruction.get("objective", {}),
        "best_candidate": reconstruction.get("best_candidate", {}),
        "nearest_true_candidate": reconstruction.get("nearest_true_candidate", {}),
    }
