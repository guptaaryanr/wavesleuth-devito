"""Budgeted challenge helpers: the game loop layer."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from .blind import blind_observed_run, challenge_secret_digest, public_world_from_secret, secret_world_hashes
from .exceptions import ValidationError
from .inversion import grid_search_circle, grid_search_ellipse
from .io import load_json, save_json, save_world
from .report import generate_html_report
from .scoring import budgeted_challenge_score, score_reconstruction
from .simulation import simulate_world
from .visualization import visualize_reconstruction, visualize_run, visualize_uncertainty, visualize_world
from .world import acquisition_preset, make_default_world, make_demo_world, validate_world

SUPPORTED_CHALLENGES = ("circle-easy", "circle-noisy", "circle-limited-angle", "circle-radius-velocity", "circle-radius-velocity-staged", "ellipse-easy")

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
        "description": "v0.3-style naive joint search over center, radius, and anomaly velocity.",
        "notes": [
            "This remains as a diagnostic baseline because joint radius/velocity search can prefer weak impostor anomalies.",
            "A low score here is a useful failure mode, not evidence that the basic center-recovery pipeline is broken.",
        ],
    },
    "circle-radius-velocity-staged": {
        "difficulty": "hard",
        "experimental": False,
        "description": "v0.4 staged center-first search over center, radius, and anomaly velocity.",
        "notes": [
            "This is the v0.4 answer to the naive joint radius/velocity failure mode.",
            "It localizes center first, keeps top-K centers, then searches radius/velocity and performs a final local center polish.",
        ],
    },
    "ellipse-easy": {
        "difficulty": "medium",
        "experimental": False,
        "description": "v0.5 first non-circle reconstruction challenge: recover the center of a rotated ellipse.",
        "notes": [
            "Known-shape challenge: the baseline holds ellipse axes, orientation, and velocity from metadata and searches the center.",
            "This is intentionally conservative: it proves non-circle inversion without turning v0.5 into a giant shape optimizer.",
            "Full unknown-ellipse discovery is left for a later staged/regularized search.",
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
        "search_strategy": "joint",
        "top_k_refine": 5,
        "final_refine_top_k": 1,
        "parameter_prior": "none",
        "radius_prior_weight": 0.0,
        "velocity_prior_weight": 0.0,
        "method": "circle-grid-search",
    }
    if challenge == "ellipse-easy":
        world = make_default_world("ellipse", name="challenge_ellipse_easy", acquisition="crossfire")
        world["grid"].update({"nx": 52, "nz": 52, "extent_x": 1.0, "extent_z": 1.0})
        world["simulation"].update(
            {
                "nt": 360,
                "dt": 0.0015,
                "space_order": 4,
                "source_frequency": 18.0,
                "shot_mode": "sequential",
                "boundary": "sponge",
                "sponge_width": 5,
                "sponge_strength": 12.0,
            }
        )
        settings.update({
            "method": "ellipse-grid-search",
            "candidate_grid_size": 5,
            "refine_levels": 1,
            "known_shape_parameters": ["radius_x", "radius_z", "angle_degrees", "anomaly_velocity"],
        })
    elif challenge == "circle-noisy":
        world["name"] = "challenge_circle_noisy"
        settings.update({"noise_level": 0.035, "amplitude_jitter": 0.035, "time_jitter": 0.0015})
    elif challenge == "circle-limited-angle":
        world["name"] = "challenge_circle_limited_angle"
        world["acquisition"] = acquisition_preset("top-only")
        world["simulation"]["shot_mode"] = "sequential"
        settings.update({"refine_levels": 1})
    elif challenge == "circle-radius-velocity":
        world["name"] = "challenge_circle_radius_velocity"
        settings.update({"search_radius": True, "search_velocity": True, "refine_levels": 0, "search_strategy": "joint"})
    elif challenge == "circle-radius-velocity-staged":
        world["name"] = "challenge_circle_radius_velocity_staged"
        settings.update({
            "search_radius": True,
            "search_velocity": True,
            "refine_levels": 1,
            "search_strategy": "staged",
            "top_k_refine": 5,
            "final_refine_top_k": 1,
            "parameter_prior": "none",
            "radius_prior_weight": 0.0,
            "velocity_prior_weight": 0.0,
        })
    else:
        world["name"] = "challenge_circle_easy"
    validate_world(world)
    return world, settings


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ("kind", "center_x", "center_z", "radius", "radius_x", "radius_z", "angle_degrees", "anomaly_velocity", "mismatch")
    compact: dict[str, Any] = {}
    for key in keys:
        if key in candidate:
            if key == "kind":
                compact[key] = str(candidate[key])
            else:
                compact[key] = _rounded(candidate[key], 6 if key == "mismatch" else 4)
    return compact




def _drop_none_leaderboard_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Remove non-applicable optional fields from leaderboard rows."""
    optional = {
        "radius_error",
        "radius_x_error",
        "radius_z_error",
        "angle_error_degrees",
        "velocity_error",
        "relative_velocity_error",
    }
    return {key: value for key, value in row.items() if not (key in optional and value is None)}

def _stable_challenge_score_from_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Return a v0.3.2-style challenge score from a saved summary.

    Older v0.3.1 summaries stored a runtime-penalized score. Leaderboards should
    display the stable v0.3.2 score when the required reconstruction and budget
    fields are available, without forcing users to rerun every challenge just to
    get the cleaned scoring formula.
    """
    stored = data.get("challenge_score", {})
    reconstruction_score = data.get("physical_score", data.get("score", {}))
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


def _backfill_velocity_score_fields(data: dict[str, Any], summary_path: Path | None = None) -> dict[str, Any]:
    """Return reconstruction score with velocity diagnostics when recoverable.

    v0.4 challenge summaries saved before v0.4.1 may not include velocity-error
    fields. When a reconstruction JSON is available, derive those fields from its
    embedded true_center and best_candidate entries so old summaries display
    cleaner leaderboards without forcing an immediate rerun.
    """
    score = data.get("physical_score", data.get("score", {}))
    result: dict[str, Any] = dict(score) if isinstance(score, dict) else {}
    if result.get("velocity_error") is not None and result.get("relative_velocity_error") is not None:
        return result

    recon: dict[str, Any] = {}
    reconstruction_path = data.get("reconstruction_path")
    if isinstance(reconstruction_path, str):
        candidates = [Path(reconstruction_path)]
        if summary_path is not None:
            candidates.append(summary_path.parent / reconstruction_path)
            candidates.append(summary_path.parent.parent / reconstruction_path)
        for path in candidates:
            try:
                if path.exists():
                    recon = load_json(path)
                    break
            except Exception:
                recon = {}

    if not recon:
        recon = data.get("reconstruction", {}) if isinstance(data.get("reconstruction"), dict) else {}

    true = recon.get("true_center", {}) if isinstance(recon.get("true_center"), dict) else {}
    best = recon.get("best_candidate", data.get("best_candidate", {}))
    if not isinstance(best, dict):
        best = {}
    try:
        true_velocity = float(true["anomaly_velocity"] if "anomaly_velocity" in true else true["velocity"])
        predicted_velocity = float(best["anomaly_velocity"] if "anomaly_velocity" in best else best["velocity"])
    except (KeyError, TypeError, ValueError):
        return result

    velocity_error = abs(predicted_velocity - true_velocity)
    relative_velocity_error = velocity_error / max(abs(true_velocity), 1.0e-12)
    result.setdefault("true_anomaly_velocity", true_velocity)
    result.setdefault("predicted_anomaly_velocity", predicted_velocity)
    result.setdefault("velocity_error", velocity_error)
    result.setdefault("anomaly_velocity_error", velocity_error)
    result.setdefault("relative_velocity_error", relative_velocity_error)
    result.setdefault("relative_anomaly_velocity_error", relative_velocity_error)
    return result


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
    targets: list[Path] = [root_path / "challenge_summary.json", root_path / "challenge_manifest.json", root_path / "challenge_score_report.json"]
    for name in SUPPORTED_CHALLENGES:
        targets.extend(
            [
                root_path / "worlds" / f"{name}.json",
                root_path / "runs" / f"{name}_obs.npz",
                root_path / "runs" / f"{name}_secret_obs.npz",
                root_path / "runs" / f"{name}_recon.json",
                root_path / "public" / f"{name}_public_world.json",
                root_path / "secret" / f"{name}_secret_world.json",
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
            root_path / "secret" / "answer_world.png",
            root_path / "secret" / "reconstruction_answer.png",
        ]
    )
    removed: list[str] = []
    for target in targets:
        item = _remove_generated_path(target, root_path)
        if item is not None:
            removed.append(item)
    return removed



def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _true_center_from_world(world: dict[str, Any]) -> dict[str, Any]:
    """Return compact true-target fields for private answer visualization."""
    anomaly = world.get("medium", {}).get("anomaly", {}) if isinstance(world.get("medium"), dict) else {}
    result: dict[str, Any] = {}
    if isinstance(anomaly, dict):
        for key in (
            "kind",
            "center_x",
            "center_z",
            "radius",
            "radius_x",
            "radius_z",
            "angle_degrees",
            "width",
            "height",
            "inner_radius",
            "outer_radius",
            "length",
        ):
            if key in anomaly:
                result[key] = anomaly[key]
    medium = world.get("medium", {})
    if isinstance(medium, dict) and "anomaly_velocity" in medium:
        result["anomaly_velocity"] = medium["anomaly_velocity"]
    return result


def private_answer_reconstruction(reconstruction: dict[str, Any], secret_world: dict[str, Any]) -> dict[str, Any]:
    """Return a private answer-view reconstruction that intentionally shows truth.

    Public blind reconstruction files may carry ``answer_hidden`` and a redacted
    public world. The private file under ``secret/reconstruction_answer.png``
    should use the secret world and explicitly clear those redaction flags.
    """
    answer = dict(reconstruction)
    answer["world"] = secret_world
    answer["answer_hidden"] = False
    answer["blind"] = False
    answer["private_answer_view"] = True
    answer["true_center"] = _true_center_from_world(secret_world)
    return answer


def score_challenge_directory(
    challenge_dir: str | Path,
    *,
    reconstruction_path: str | Path | None = None,
    update_reconstruction: bool = False,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score a challenge directory using its secret answer when available."""
    root = Path(challenge_dir)
    summary_path = root / "challenge_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    challenge_name = str(summary.get("challenge") or root.name)

    if reconstruction_path is None:
        raw_recon = summary.get("reconstruction_path")
        if isinstance(raw_recon, str):
            recon_path = Path(raw_recon)
            if not recon_path.exists():
                recon_path = root / raw_recon
        else:
            recon_path = root / "runs" / f"{challenge_name}_recon.json"
    else:
        recon_path = Path(reconstruction_path)
    if not recon_path.exists():
        raise ValidationError(f"Reconstruction file not found: {recon_path}")

    secret_candidates = sorted((root / "secret").glob("*_secret_world.json"))
    secret_path = _first_existing(secret_candidates)
    if secret_path is None:
        world_candidates: list[Path] = []
        raw_secret = summary.get("secret_world_path")
        raw_world = summary.get("world_path")
        for raw in (raw_secret, raw_world):
            if isinstance(raw, str):
                p = Path(raw)
                world_candidates.extend([p, root / raw])
        world_candidates.extend(sorted((root / "worlds").glob("*.json")))
        secret_path = _first_existing(world_candidates)
    if secret_path is None:
        raise ValidationError(f"No secret/open world file found under {root}.")

    true_world = load_json(secret_path)
    secret_hashes = secret_world_hashes(true_world, secret_path)
    reconstruction = load_json(recon_path)
    physical_score = score_reconstruction(true_world, reconstruction)
    stored = summary.get("challenge_score", {}) if isinstance(summary.get("challenge_score"), dict) else {}
    n_forward_runs = int(stored.get("n_forward_runs", reconstruction.get("candidate_grid", {}).get("forward_runs", len(reconstruction.get("candidates", [])))))
    n_sources = int(stored.get("n_sources", len(true_world.get("acquisition", {}).get("sources", []))))
    n_receivers = int(stored.get("n_receivers", len(true_world.get("acquisition", {}).get("receivers", []))))
    challenge_score = budgeted_challenge_score(
        physical_score,
        n_forward_runs=n_forward_runs,
        n_sources=n_sources,
        n_receivers=n_receivers,
        runtime_seconds=summary.get("runtime_seconds"),
    )
    result = {
        "schema_version": "0.6.1",
        "challenge": summary.get("challenge", challenge_name),
        "blind": bool(summary.get("blind", False)),
        "secret_world_path": str(secret_path),
        "secret_world_sha256": secret_hashes["secret_world_sha256"],
        "secret_world_file_sha256": secret_hashes["secret_world_file_sha256"],
        "secret_world_canonical_sha256": secret_hashes["secret_world_canonical_sha256"],
        "reconstruction_path": str(recon_path),
        "physical_score": physical_score,
        "challenge_score": challenge_score,
        "score": physical_score,
    }
    if update_reconstruction:
        reconstruction["schema_version"] = "0.6.1"
        reconstruction["physical_score"] = physical_score
        reconstruction["score"] = physical_score
        save_json(reconstruction, recon_path)
        result["updated_reconstruction"] = True
    else:
        result["updated_reconstruction"] = False
    if out_path is not None:
        save_json(result, out_path)
    return result


def run_challenge(
    challenge: str,
    *,
    out_dir: str | Path,
    candidate_grid_size: int | None = None,
    refine_levels: int | None = None,
    quiet: bool = False,
    clean: bool = True,
    blind: bool = False,
) -> dict[str, Any]:
    """Run a named challenge and write a summary JSON."""
    root = Path(out_dir)
    cleaned_paths: list[str] = clean_challenge_output(root) if clean else []
    worlds = root / "worlds"
    runs = root / "runs"
    figures = root / "figures"
    reports = root / "reports"
    public_dir = root / "public"
    secret_dir = root / "secret"
    directories = (worlds, runs, figures, reports, public_dir, secret_dir) if blind else (worlds, runs, figures, reports)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    world, settings = make_challenge_world(challenge)
    meta = CHALLENGE_METADATA[challenge]
    if candidate_grid_size is not None:
        settings["candidate_grid_size"] = int(candidate_grid_size)
    if refine_levels is not None:
        settings["refine_levels"] = int(refine_levels)

    public_world = public_world_from_secret(world, challenge=challenge) if blind else world
    world_path = worlds / f"{challenge}.json"
    public_world_path = public_dir / f"{challenge}_public_world.json" if blind else world_path
    secret_world_path = secret_dir / f"{challenge}_secret_world.json" if blind else None
    manifest_path = root / "challenge_manifest.json"
    run_path = runs / f"{challenge}_obs.npz"
    secret_run_path = runs / f"{challenge}_secret_obs.npz"
    recon_path = runs / f"{challenge}_recon.json"
    save_world(public_world, world_path)
    if blind:
        save_world(public_world, public_world_path)
        save_world(world, secret_world_path)

    t0 = perf_counter()
    simulation_run_path = secret_run_path if blind else run_path
    simulate_world(
        world,
        out_path=str(simulation_run_path),
        save_wavefield=False,
        quiet=quiet,
        shot_mode=world["simulation"].get("shot_mode", "sequential"),
        noise_level=float(settings["noise_level"]),
        receiver_dropout=float(settings["receiver_dropout"]),
        amplitude_jitter=float(settings["amplitude_jitter"]),
        time_jitter=float(settings["time_jitter"]),
    )
    if blind:
        blind_observed_run(secret_run_path, run_path, public_world)
        try:
            secret_run_path.unlink()
        except OSError:
            pass
    if settings.get("method") == "ellipse-grid-search":
        reconstruction = grid_search_ellipse(
            run_path,
            out_path=recon_path,
            candidate_grid_size=int(settings["candidate_grid_size"]),
            refine_levels=int(settings["refine_levels"]),
            mismatch_mode=str(settings["mismatch_mode"]),
            metric=str(settings["metric"]),
            quiet=quiet,
        )
    else:
        reconstruction = grid_search_circle(
            run_path,
            out_path=recon_path,
            candidate_grid_size=int(settings["candidate_grid_size"]),
            refine_levels=int(settings["refine_levels"]),
            mismatch_mode=str(settings["mismatch_mode"]),
            metric=str(settings["metric"]),
            search_radius=bool(settings.get("search_radius", False)),
            search_velocity=bool(settings.get("search_velocity", False)),
            search_strategy=str(settings.get("search_strategy", "joint")),
            top_k_refine=int(settings.get("top_k_refine", 5)),
            final_refine_top_k=int(settings.get("final_refine_top_k", 1)),
            parameter_prior=str(settings.get("parameter_prior", "none")),
            radius_prior_weight=float(settings.get("radius_prior_weight", 0.0)),
            velocity_prior_weight=float(settings.get("velocity_prior_weight", 0.0)),
            quiet=quiet,
        )
    runtime = perf_counter() - t0

    visualize_world(public_world, figures / "world.png")
    if blind:
        visualize_world(world, secret_dir / "answer_world.png")
    visualize_run(run_path, figures / "traces.png")
    visualize_reconstruction(reconstruction, figures / "reconstruction.png")
    if blind:
        visualize_reconstruction(private_answer_reconstruction(reconstruction, world), secret_dir / "reconstruction_answer.png")
    visualize_uncertainty(reconstruction, figures / "uncertainty.png")
    report_path = generate_html_report(recon_path, reports / "report.html")

    score = score_reconstruction(world, reconstruction)
    secret_hashes = secret_world_hashes(world, secret_world_path) if blind else {
        "secret_world_sha256": None,
        "secret_world_file_sha256": None,
        "secret_world_canonical_sha256": None,
    }
    n_forward_runs = int(reconstruction.get("candidate_grid", {}).get("forward_runs", len(reconstruction.get("candidates", []))))
    challenge_score = budgeted_challenge_score(
        score,
        n_forward_runs=n_forward_runs,
        n_sources=len(world["acquisition"]["sources"]),
        n_receivers=len(world["acquisition"]["receivers"]),
        runtime_seconds=runtime,
    )
    secret_digest = secret_hashes["secret_world_sha256"]
    manifest = {
        "schema_version": "0.6.1",
        "challenge": challenge,
        "difficulty": meta["difficulty"],
        "experimental": bool(meta["experimental"]),
        "blind": bool(blind),
        "public_world_path": str(public_world_path),
        "observed_run_path": str(run_path),
        "suggested_reconstruction_path": str(recon_path),
        "secret_world_sha256": secret_hashes["secret_world_sha256"],
        "secret_world_file_sha256": secret_hashes["secret_world_file_sha256"],
        "secret_world_canonical_sha256": secret_hashes["secret_world_canonical_sha256"],
        "hash_notes": [
            "secret_world_sha256 and secret_world_file_sha256 are file-byte hashes of the saved secret world JSON when blind=True.",
            "secret_world_canonical_sha256 hashes compact sorted-key JSON and is stable under formatting changes.",
        ],
        "shareable_public_files": [str(public_world_path), str(run_path), str(manifest_path)],
        "notes": [
            "Blind challenge: share public files only, not the secret directory." if blind else "Open challenge: world metadata includes the answer.",
            "Known-shape hints may remain public for current baseline inversions.",
        ],
    }
    save_json(manifest, manifest_path)

    summary = {
        "schema_version": "0.6.1",
        "challenge": challenge,
        "difficulty": meta["difficulty"],
        "experimental": bool(meta["experimental"]),
        "description": meta["description"],
        "notes": list(meta["notes"]),
        "settings": settings,
        "blind": bool(blind),
        "manifest_path": str(manifest_path),
        "public_world_path": str(public_world_path),
        "secret_world_path": None if secret_world_path is None else str(secret_world_path),
        "secret_world_sha256": secret_hashes["secret_world_sha256"],
        "secret_world_file_sha256": secret_hashes["secret_world_file_sha256"],
        "secret_world_canonical_sha256": secret_hashes["secret_world_canonical_sha256"],
        "world_path": str(world_path),
        "run_path": str(run_path),
        "reconstruction_path": str(recon_path),
        "figures_dir": str(figures),
        "answer_world_figure": str(secret_dir / "answer_world.png") if blind else None,
        "answer_reconstruction_figure": str(secret_dir / "reconstruction_answer.png") if blind else None,
        "report_path": str(report_path),
        "physical_score": score,
        "score": score,
        "challenge_score": challenge_score,
        "score_summary": {
            "score": _rounded(challenge_score.get("score"), 3),
            "iou": _rounded(score.get("iou"), 3),
            "center_error": _rounded(score.get("center_error"), 4),
            "normalized_center_error": _rounded(score.get("normalized_center_error"), 4),
            "radius_error": _rounded(score.get("radius_error"), 4),
            "radius_x_error": _rounded(score.get("radius_x_error"), 4),
            "radius_z_error": _rounded(score.get("radius_z_error"), 4),
            "angle_error_degrees": _rounded(score.get("angle_error_degrees"), 3),
            "velocity_error": _rounded(score.get("velocity_error"), 4),
            "relative_velocity_error": _rounded(score.get("relative_velocity_error"), 4),
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
            reconstruction_score = _backfill_velocity_score_fields(data, candidate)
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
                "blind": bool(data.get("blind", False)),
                "score": rounded_score,
                "iou": _rounded(reconstruction_score.get("iou"), 3),
                "center_error": _rounded(reconstruction_score.get("center_error"), 4),
                "normalized_center_error": _rounded(reconstruction_score.get("normalized_center_error"), 4),
                "radius_error": _rounded(reconstruction_score.get("radius_error"), 4),
                "radius_x_error": _rounded(reconstruction_score.get("radius_x_error"), 4),
                "radius_z_error": _rounded(reconstruction_score.get("radius_z_error"), 4),
                "angle_error_degrees": _rounded(reconstruction_score.get("angle_error_degrees"), 3),
                "velocity_error": _rounded(reconstruction_score.get("velocity_error"), 4),
                "relative_velocity_error": _rounded(reconstruction_score.get("relative_velocity_error"), 4),
                "forward_runs": score.get("n_forward_runs"),
                "runtime_seconds": _rounded(data.get("runtime_seconds"), 3),
                "best_candidate": data.get("best_candidate_summary") or _compact_candidate(data.get("best_candidate", {})),
            }
            row = _drop_none_leaderboard_fields(row)
            sort_score = rounded_score if rounded_score is not None else float("-inf")
            sort_iou = row["iou"] if row["iou"] is not None else float("-inf")
            sort_center = row["center_error"] if row["center_error"] is not None else float("inf")
            row["_sort_key"] = (rounded_score is None, -float(sort_score), -float(sort_iou), float(sort_center), int(row["forward_runs"] or 10**9), str(row["challenge"]))
            rows.append(row)
    rows.sort(key=lambda row: row["_sort_key"])
    for row in rows:
        row.pop("_sort_key", None)
    return rows
