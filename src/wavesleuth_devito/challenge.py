"""Budgeted challenge helpers: the game loop layer."""

from __future__ import annotations

import math
import shutil
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

CHALLENGE_METADATA: dict[str, dict[str, Any]] = {
    "circle-easy": {
        "difficulty": "easy",
        "experimental": False,
        "description": "Crossfire circular-anomaly reconstruction with clean observations.",
        "notes": ["Good baseline for checking center recovery and score stability."],
    },
    "circle-noisy": {
        "difficulty": "medium",
        "experimental": False,
        "description": "Same hidden circle with mild deterministic noise and timing/amplitude perturbations.",
        "notes": ["Currently mild enough that differential crossfire inversion may tie the clean case."],
    },
    "circle-limited-angle": {
        "difficulty": "medium",
        "experimental": False,
        "description": "Limited-angle top-only acquisition for the same circular target.",
        "notes": ["Expected to be less certain than crossfire because the illumination is less diverse."],
    },
    "circle-radius-velocity": {
        "difficulty": "hard",
        "experimental": True,
        "description": "Searches center, radius, and anomaly velocity with a naive joint grid objective.",
        "notes": [
            "This is intentionally marked experimental because joint radius/velocity search can prefer weak impostor anomalies.",
            "A low score here is a useful failure mode, not evidence that the basic center-recovery pipeline is broken.",
        ],
    },
}


def make_challenge_world(challenge: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return `(world, settings)` for a named challenge."""
    if challenge not in SUPPORTED_CHALLENGES:
        raise ValidationError(f"Unsupported challenge {challenge!r}. Supported: {', '.join(SUPPORTED_CHALLENGES)}")
    world = make_demo_world()
    meta = CHALLENGE_METADATA[challenge]
    settings: dict[str, Any] = {
        "challenge": challenge,
        "difficulty": meta["difficulty"],
        "experimental": bool(meta["experimental"]),
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


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ("center_x", "center_z", "radius", "anomaly_velocity", "mismatch")
    compact: dict[str, Any] = {}
    for key in keys:
        if key in candidate:
            compact[key] = _rounded(candidate[key], 6 if key == "mismatch" else 4)
    return compact


def _stable_challenge_score_from_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Return a v0.3.2-style challenge score from a saved summary.

    Older v0.3.1 summaries stored a runtime-penalized score. Leaderboards should
    display the stable v0.3.2 score when the required reconstruction and budget
    fields are available, without forcing users to rerun every challenge just to
    get the cleaned scoring formula.
    """
    stored = data.get("challenge_score", {})
    reconstruction_score = data.get("score", {})
    if not isinstance(stored, dict) or not isinstance(reconstruction_score, dict):
        return stored if isinstance(stored, dict) else {}
    required = ("n_forward_runs", "n_sources", "n_receivers")
    if reconstruction_score.get("supported", False) and all(key in stored for key in required):
        try:
            return budgeted_challenge_score(
                reconstruction_score,
                n_forward_runs=int(stored["n_forward_runs"]),
                n_sources=int(stored["n_sources"]),
                n_receivers=int(stored["n_receivers"]),
                runtime_seconds=data.get("runtime_seconds", stored.get("runtime_seconds")),
            )
        except (TypeError, ValueError):
            return stored
    return stored


def _rounded(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _remove_generated_path(path: Path, root: Path) -> str | None:
    """Remove one known generated challenge artifact, returning its relative path."""
    if not path.exists():
        return None
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def clean_challenge_output(root: str | Path) -> list[str]:
    """Remove challenge-owned artifacts from an output directory.

    The cleanup is intentionally narrow. It removes files that WaveSleuth's
    challenge command is known to generate, while leaving unrelated user files
    alone. This keeps reruns deterministic and prevents stale files from an old
    challenge from lingering beside a fresh one.
    """
    root_path = Path(root)
    targets: list[Path] = [root_path / "challenge_summary.json"]
    for name in SUPPORTED_CHALLENGES:
        targets.extend(
            [
                root_path / "worlds" / f"{name}.json",
                root_path / "runs" / f"{name}_obs.npz",
                root_path / "runs" / f"{name}_recon.json",
            ]
        )
    targets.extend(
        [
            root_path / "figures" / "world.png",
            root_path / "figures" / "traces.png",
            root_path / "figures" / "reconstruction.png",
            root_path / "figures" / "uncertainty.png",
            root_path / "reports" / "report.html",
            root_path / "reports" / "report_assets",
        ]
    )
    removed: list[str] = []
    for target in targets:
        item = _remove_generated_path(target, root_path)
        if item is not None:
            removed.append(item)
    return removed


def run_challenge(
    challenge: str,
    *,
    out_dir: str | Path,
    candidate_grid_size: int | None = None,
    refine_levels: int | None = None,
    quiet: bool = False,
    clean: bool = True,
) -> dict[str, Any]:
    """Run a named challenge and write a summary JSON."""
    root = Path(out_dir)
    cleaned_paths: list[str] = clean_challenge_output(root) if clean else []
    worlds = root / "worlds"
    runs = root / "runs"
    figures = root / "figures"
    reports = root / "reports"
    for directory in (worlds, runs, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)

    world, settings = make_challenge_world(challenge)
    meta = CHALLENGE_METADATA[challenge]
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
        "difficulty": meta["difficulty"],
        "experimental": bool(meta["experimental"]),
        "description": meta["description"],
        "notes": list(meta["notes"]),
        "settings": settings,
        "world_path": str(world_path),
        "run_path": str(run_path),
        "reconstruction_path": str(recon_path),
        "figures_dir": str(figures),
        "report_path": str(report_path),
        "score": score,
        "challenge_score": challenge_score,
        "score_summary": {
            "score": _rounded(challenge_score.get("score"), 3),
            "iou": _rounded(score.get("iou"), 3),
            "center_error": _rounded(score.get("center_error"), 4),
            "normalized_center_error": _rounded(score.get("normalized_center_error"), 4),
            "radius_error": _rounded(score.get("radius_error"), 4),
            "forward_runs": n_forward_runs,
        },
        "best_candidate": reconstruction.get("best_candidate", {}),
        "best_candidate_summary": _compact_candidate(reconstruction.get("best_candidate", {})),
        "objective": reconstruction.get("objective", {}),
        "search": reconstruction.get("search", {}),
        "candidate_grid": reconstruction.get("candidate_grid", {}),
        "uncertainty": reconstruction.get("uncertainty", {}),
        "runtime_seconds": runtime,
        "cleaned_before_run": bool(clean),
        "cleaned_paths": cleaned_paths,
    }
    save_json(summary, root / "challenge_summary.json")
    return summary


def collect_leaderboard(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Collect challenge summaries under a list of files or directories.

    Displayed scores are rounded before sorting so tiny runtime jitter does not
    make two practically tied runs look meaningfully ordered.
    """
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
            score = _stable_challenge_score_from_summary(data)
            reconstruction_score = data.get("score", {})
            supported = bool(score.get("supported", False))
            raw_score = float(score.get("score", float("nan"))) if supported else float("nan")
            rounded_score = _rounded(raw_score, 3)
            challenge_name = str(data.get("challenge"))
            fallback_meta = CHALLENGE_METADATA.get(challenge_name, {})
            row = {
                "path": str(candidate),
                "challenge": data.get("challenge"),
                "difficulty": data.get("difficulty") or data.get("settings", {}).get("difficulty") or fallback_meta.get("difficulty"),
                "experimental": bool(data.get("experimental", data.get("settings", {}).get("experimental", fallback_meta.get("experimental", False)))),
                "score": rounded_score,
                "iou": _rounded(reconstruction_score.get("iou"), 3),
                "center_error": _rounded(reconstruction_score.get("center_error"), 4),
                "normalized_center_error": _rounded(reconstruction_score.get("normalized_center_error"), 4),
                "radius_error": _rounded(reconstruction_score.get("radius_error"), 4),
                "forward_runs": score.get("n_forward_runs"),
                "runtime_seconds": _rounded(data.get("runtime_seconds"), 3),
                "best_candidate": data.get("best_candidate_summary") or _compact_candidate(data.get("best_candidate", {})),
            }
            sort_score = rounded_score if rounded_score is not None else float("-inf")
            sort_iou = row["iou"] if row["iou"] is not None else float("-inf")
            sort_center = row["center_error"] if row["center_error"] is not None else float("inf")
            row["_sort_key"] = (rounded_score is None, -float(sort_score), -float(sort_iou), float(sort_center), int(row["forward_runs"] or 10**9), str(row["challenge"]))
            rows.append(row)
    rows.sort(key=lambda row: row["_sort_key"])
    for row in rows:
        row.pop("_sort_key", None)
    return rows
