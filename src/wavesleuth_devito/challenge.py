"""Budgeted challenge helpers: the game loop layer."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from .exceptions import ValidationError
from .inversion import grid_search_circle
from .io import load_json, save_json, save_world
from .report import generate_html_report
from .scoring import budgeted_challenge_score, score_reconstruction
from .simulation import simulate_world
from .visualization import visualize_reconstruction, visualize_run, visualize_uncertainty, visualize_world
from .world import acquisition_preset, make_demo_world, validate_world

SUPPORTED_CHALLENGES = ("circle-easy", "circle-noisy", "circle-limited-angle", "circle-radius-velocity")


def make_challenge_world(challenge: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return `(world, settings)` for a named challenge."""
    if challenge not in SUPPORTED_CHALLENGES:
        raise ValidationError(f"Unsupported challenge {challenge!r}. Supported: {', '.join(SUPPORTED_CHALLENGES)}")
    world = make_demo_world()
    settings: dict[str, Any] = {
        "challenge": challenge,
        "candidate_grid_size": 5,
        "refine_levels": 1,
        "mismatch_mode": "differential",
        "metric": "l2",
        "noise_level": 0.0,
        "receiver_dropout": 0.0,
        "amplitude_jitter": 0.0,
        "time_jitter": 0.0,
        "search_radius": False,
        "search_velocity": False,
    }
    if challenge == "circle-noisy":
        world["name"] = "challenge_circle_noisy"
        settings.update({"noise_level": 0.035, "amplitude_jitter": 0.035, "time_jitter": 0.0015})
    elif challenge == "circle-limited-angle":
        world["name"] = "challenge_circle_limited_angle"
        world["acquisition"] = acquisition_preset("top-only")
        world["simulation"]["shot_mode"] = "sequential"
        settings.update({"refine_levels": 1})
    elif challenge == "circle-radius-velocity":
        world["name"] = "challenge_circle_radius_velocity"
        settings.update({"search_radius": True, "search_velocity": True, "refine_levels": 0})
    else:
        world["name"] = "challenge_circle_easy"
    validate_world(world)
    return world, settings


def run_challenge(
    challenge: str,
    *,
    out_dir: str | Path,
    candidate_grid_size: int | None = None,
    refine_levels: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run a named challenge and write a summary JSON."""
    root = Path(out_dir)
    worlds = root / "worlds"
    runs = root / "runs"
    figures = root / "figures"
    reports = root / "reports"
    for directory in (worlds, runs, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)

    world, settings = make_challenge_world(challenge)
    if candidate_grid_size is not None:
        settings["candidate_grid_size"] = int(candidate_grid_size)
    if refine_levels is not None:
        settings["refine_levels"] = int(refine_levels)

    world_path = worlds / f"{challenge}.json"
    run_path = runs / f"{challenge}_obs.npz"
    recon_path = runs / f"{challenge}_recon.json"
    save_world(world, world_path)

    t0 = perf_counter()
    simulate_world(
        world,
        out_path=str(run_path),
        save_wavefield=False,
        quiet=quiet,
        shot_mode=world["simulation"].get("shot_mode", "sequential"),
        noise_level=float(settings["noise_level"]),
        receiver_dropout=float(settings["receiver_dropout"]),
        amplitude_jitter=float(settings["amplitude_jitter"]),
        time_jitter=float(settings["time_jitter"]),
    )
    reconstruction = grid_search_circle(
        run_path,
        out_path=recon_path,
        candidate_grid_size=int(settings["candidate_grid_size"]),
        refine_levels=int(settings["refine_levels"]),
        mismatch_mode=str(settings["mismatch_mode"]),
        metric=str(settings["metric"]),
        search_radius=bool(settings.get("search_radius", False)),
        search_velocity=bool(settings.get("search_velocity", False)),
        quiet=quiet,
    )
    runtime = perf_counter() - t0

    visualize_world(world, figures / "world.png")
    visualize_run(run_path, figures / "traces.png")
    visualize_reconstruction(reconstruction, figures / "reconstruction.png")
    visualize_uncertainty(reconstruction, figures / "uncertainty.png")
    report_path = generate_html_report(recon_path, reports / "report.html")

    score = score_reconstruction(world, reconstruction)
    n_forward_runs = int(reconstruction.get("candidate_grid", {}).get("forward_runs", len(reconstruction.get("candidates", []))))
    challenge_score = budgeted_challenge_score(
        score,
        n_forward_runs=n_forward_runs,
        n_sources=len(world["acquisition"]["sources"]),
        n_receivers=len(world["acquisition"]["receivers"]),
        runtime_seconds=runtime,
    )
    summary = {
        "challenge": challenge,
        "settings": settings,
        "world_path": str(world_path),
        "run_path": str(run_path),
        "reconstruction_path": str(recon_path),
        "figures_dir": str(figures),
        "report_path": str(report_path),
        "score": score,
        "challenge_score": challenge_score,
        "best_candidate": reconstruction.get("best_candidate", {}),
        "runtime_seconds": runtime,
    }
    save_json(summary, root / "challenge_summary.json")
    return summary


def collect_leaderboard(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Collect challenge summaries under a list of files or directories."""
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        candidates: list[Path]
        if path.is_dir():
            candidates = sorted(path.rglob("challenge_summary.json"))
        else:
            candidates = [path]
        for candidate in candidates:
            if not candidate.exists():
                continue
            data = load_json(candidate)
            score = data.get("challenge_score", {})
            rows.append(
                {
                    "path": str(candidate),
                    "challenge": data.get("challenge"),
                    "score": float(score.get("score", float("nan"))) if score.get("supported", False) else float("nan"),
                    "iou": data.get("score", {}).get("iou"),
                    "center_error": data.get("score", {}).get("center_error"),
                    "forward_runs": score.get("n_forward_runs"),
                    "runtime_seconds": data.get("runtime_seconds"),
                }
            )
    rows.sort(key=lambda row: (row["score"] != row["score"], -row["score"] if row["score"] == row["score"] else 0.0))
    return rows
