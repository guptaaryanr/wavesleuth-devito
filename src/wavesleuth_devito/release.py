"""Release-hardening helpers for WaveSleuth-Devito v0.9.

This module is intentionally lightweight. It validates artifacts, runs the
standard challenge suite, generates compact release reports, and provides a
small environment doctor. It does not change solver or inversion numerics.
"""

from __future__ import annotations

import html
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .exceptions import ValidationError
from .io import load_json, load_run_npz, world_from_run, save_json, ensure_parent
from .metadata import PROJECT_NAME, __version__, base_metadata
from .world import validate_world, velocity_model_from_world, anomaly_kind

RELEASE_SCHEMA_VERSION = "0.9.0"
DEFAULT_RELEASE_CHALLENGES = (
    "circle-easy",
    "ellipse-easy",
    "circle-radius-velocity-staged",
    "mask-cell-easy",
)


def version_tuple(value: str) -> tuple[int, int, int]:
    """Parse a simple semantic version into a comparable triple."""
    parts = value.split(".")[:3]
    out: list[int] = []
    for part in parts:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])  # type: ignore[return-value]


def _status(ok: bool, *, kind: str, path: str | None = None, errors: list[str] | None = None, warnings: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": bool(ok),
        "kind": kind,
        "path": path,
        "errors": list(errors or []),
        "warnings": list(warnings or []),
    }
    result.update(extra)
    return result


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def doctor_report(*, try_devito: bool = False) -> dict[str, Any]:
    """Return a compact environment report for debugging release checks."""
    required = ["numpy", "matplotlib"]
    optional = ["pytest", "devito"]
    modules = {name: _module_available(name) for name in required + optional}
    devito_import: dict[str, Any] | None = None
    if try_devito:
        try:
            import devito  # type: ignore

            devito_import = {"ok": True, "version": getattr(devito, "__version__", "unknown")}
        except Exception as exc:  # pragma: no cover - environment-specific
            devito_import = {"ok": False, "error": str(exc)}
    return {
        **base_metadata(),
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "package": {"name": PROJECT_NAME, "version": __version__, "version_tuple": list(version_tuple(__version__))},
        "modules": modules,
        "devito_import": devito_import,
        "notes": [
            "Devito is only required for simulation/inversion commands.",
            "v0.9 doctor does not run a numerical simulation unless future checks explicitly add one.",
        ],
    }


def normalize_challenge_summary_schema(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a v0.9-friendly challenge summary without mutating input.

    Earlier versions used `score` for the physical reconstruction score and
    `challenge_score` for the budgeted game score. v0.9 keeps those aliases for
    backward compatibility but encourages `physical_score` and `challenge_score`.
    """
    data = json.loads(json.dumps(summary))
    physical = data.get("physical_score")
    legacy_score = data.get("score")
    if not isinstance(physical, dict) and isinstance(legacy_score, dict) and "iou" in legacy_score:
        physical = legacy_score
    if not isinstance(physical, dict):
        physical = {}
    challenge_score = data.get("challenge_score")
    if not isinstance(challenge_score, dict):
        if isinstance(legacy_score, dict) and "score" in legacy_score and "iou" not in legacy_score:
            challenge_score = legacy_score
        else:
            challenge_score = {}
    data["physical_score"] = physical
    data["challenge_score"] = challenge_score
    data["schema_version"] = str(data.get("schema_version") or RELEASE_SCHEMA_VERSION)
    data["schema_notes"] = list(data.get("schema_notes", []))
    note = "v0.9 normalized aliases: physical_score is reconstruction quality; challenge_score is budgeted game score."
    if note not in data["schema_notes"]:
        data["schema_notes"].append(note)
    return data


def _resolve_reference(base: Path, raw: Any) -> Path | None:
    if raw is None:
        return None
    p = Path(str(raw))
    candidates = [p]
    if not p.is_absolute():
        candidates.append(base / p)
        candidates.append(base / p.name)
        candidates.append(base.parent / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_world_file(path: str | Path) -> dict[str, Any]:
    """Validate a world JSON file."""
    p = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    extra: dict[str, Any] = {}
    try:
        world = load_json(p)
        validate_world(world)
        model = velocity_model_from_world(world)
        extra = {
            "name": world.get("name"),
            "world_kind": anomaly_kind(world),
            "grid_shape": list(model.shape),
            "velocity_min": float(np.min(model)),
            "velocity_max": float(np.max(model)),
        }
    except Exception as exc:
        errors.append(str(exc))
    return _status(not errors, kind="world", path=str(p), errors=errors, warnings=warnings, **extra)


def validate_run_file(path: str | Path) -> dict[str, Any]:
    """Validate a simulation `.npz` run file."""
    p = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    extra: dict[str, Any] = {}
    try:
        run = load_run_npz(p)
        traces = np.asarray(run["receiver_traces"])
        if traces.ndim not in (2, 3):
            errors.append(f"receiver_traces must be 2D or 3D, got shape {traces.shape}")
        time = np.asarray(run["time"])
        if time.ndim != 1:
            errors.append(f"time must be 1D, got shape {time.shape}")
        world = world_from_run(run)
        model = np.asarray(run["velocity_model"])
        expected = (int(world["grid"]["nx"]), int(world["grid"]["nz"]))
        if model.size and model.shape != expected:
            errors.append(f"velocity_model shape {model.shape} does not match world grid {expected}")
        if traces.ndim == 2:
            layout = "time_receiver"
            n_shots = 1
            nt = int(traces.shape[0])
            n_receivers = int(traces.shape[1])
        elif traces.ndim == 3:
            layout = "shot_time_receiver"
            n_shots = int(traces.shape[0])
            nt = int(traces.shape[1])
            n_receivers = int(traces.shape[2])
        else:
            layout = "unknown"
            n_shots = None
            nt = None
            n_receivers = None
        extra = {
            "world_name": world.get("name"),
            "world_kind": anomaly_kind(world),
            "trace_shape": list(traces.shape),
            "trace_layout": layout,
            "n_shots": n_shots,
            "nt": nt,
            "n_receivers": n_receivers,
            "time_samples": int(time.shape[0]) if time.ndim == 1 else None,
        }
    except Exception as exc:
        errors.append(str(exc))
    return _status(not errors, kind="run", path=str(p), errors=errors, warnings=warnings, **extra)


def validate_reconstruction_file(path: str | Path) -> dict[str, Any]:
    """Validate a reconstruction JSON file at the schema level."""
    p = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    extra: dict[str, Any] = {}
    try:
        recon = load_json(p)
        if not isinstance(recon.get("best_candidate"), dict):
            errors.append("missing best_candidate dictionary")
        method = recon.get("method")
        if not method:
            warnings.append("missing method field")
        physical = recon.get("physical_score") or recon.get("score")
        if isinstance(physical, dict):
            extra["iou"] = physical.get("iou")
            extra["center_error"] = physical.get("center_error")
        if not isinstance(physical, dict):
            warnings.append("missing physical_score/score dictionary")
        extra.update(
            {
                "method": method,
                "target_kind": recon.get("target_kind") or (recon.get("best_candidate", {}) or {}).get("kind"),
                "best_candidate_keys": sorted((recon.get("best_candidate", {}) or {}).keys()),
                "forward_runs": (recon.get("candidate_grid", {}) or {}).get("forward_runs"),
            }
        )
    except Exception as exc:
        errors.append(str(exc))
    return _status(not errors, kind="reconstruction", path=str(p), errors=errors, warnings=warnings, **extra)


def validate_challenge_directory(path: str | Path, *, strict: bool = False) -> dict[str, Any]:
    """Validate a challenge output directory or challenge_summary.json file."""
    raw = Path(path)
    base = raw if raw.is_dir() else raw.parent
    summary_path = raw / "challenge_summary.json" if raw.is_dir() else raw
    errors: list[str] = []
    warnings: list[str] = []
    extra: dict[str, Any] = {}
    if not summary_path.exists():
        return _status(False, kind="challenge", path=str(raw), errors=[f"missing challenge summary: {summary_path}"], warnings=[])
    try:
        summary = normalize_challenge_summary_schema(load_json(summary_path))
        challenge = str(summary.get("challenge", "unknown"))
        blind = bool(summary.get("blind", False))
        extra.update(
            {
                "challenge": challenge,
                "blind": blind,
                "difficulty": summary.get("difficulty"),
                "physical_iou": summary.get("physical_score", {}).get("iou"),
                "challenge_score": summary.get("challenge_score", {}).get("score"),
            }
        )
        for key in ("run_path", "reconstruction_path", "world_path"):
            ref = _resolve_reference(base, summary.get(key))
            if ref is None or not ref.exists():
                errors.append(f"missing referenced {key}: {summary.get(key)!r}")
        manifest_ref = _resolve_reference(base, summary.get("manifest_path") or base / "challenge_manifest.json")
        if manifest_ref is None or not manifest_ref.exists():
            warnings.append("missing challenge manifest")
        else:
            manifest = load_json(manifest_ref)
            manifest_version = str(manifest.get("schema_version", ""))
            extra["manifest_schema_version"] = manifest_version
            if manifest_version.startswith("0.6"):
                warnings.append("manifest schema_version is historical; v0.9 can read it but new outputs should use 0.9.0")
            if blind:
                secret_ref = _resolve_reference(base, summary.get("secret_world_path") or manifest.get("secret_world_path"))
                expected_hash = manifest.get("secret_world_file_sha256") or manifest.get("secret_world_sha256")
                if secret_ref is not None and secret_ref.exists() and expected_hash:
                    import hashlib

                    actual = hashlib.sha256(secret_ref.read_bytes()).hexdigest()
                    extra["secret_world_file_sha256_matches"] = actual == expected_hash
                    if actual != expected_hash:
                        errors.append("secret world file-byte hash does not match manifest")
                elif strict:
                    errors.append("blind challenge is missing secret hash verification inputs")
        score = summary.get("challenge_score", {})
        if not isinstance(score, dict) or "score" not in score:
            warnings.append("missing challenge_score.score")
        physical = summary.get("physical_score", {})
        if not isinstance(physical, dict) or "iou" not in physical:
            warnings.append("missing physical_score.iou")
    except Exception as exc:
        errors.append(str(exc))
    return _status(not errors, kind="challenge", path=str(raw), errors=errors, warnings=warnings, **extra)


def validate_active_directory(path: str | Path) -> dict[str, Any]:
    """Validate an active-demo output directory or active_summary.json file."""
    raw = Path(path)
    summary_path = raw / "active_summary.json" if raw.is_dir() else raw
    errors: list[str] = []
    warnings: list[str] = []
    extra: dict[str, Any] = {}
    if not summary_path.exists():
        return _status(False, kind="active", path=str(raw), errors=[f"missing active summary: {summary_path}"], warnings=[])
    try:
        summary = load_json(summary_path)
        rounds = summary.get("rounds", [])
        if not isinstance(rounds, list) or not rounds:
            errors.append("active_summary has no rounds")
        extra.update(
            {
                "strategy": summary.get("strategy"),
                "kind": summary.get("kind"),
                "round_count": len(rounds) if isinstance(rounds, list) else None,
                "initial_reconstruction_score": summary.get("initial_reconstruction_score"),
                "final_reconstruction_score": summary.get("final_reconstruction_score"),
                "score_delta": summary.get("score_delta"),
            }
        )
        for idx, round_data in enumerate(rounds if isinstance(rounds, list) else []):
            shape = round_data.get("trace_shape")
            if shape is not None and isinstance(shape, list) and len(shape) != 3:
                warnings.append(f"round {idx + 1} trace_shape is not standardized 3D: {shape}")
    except Exception as exc:
        errors.append(str(exc))
    return _status(not errors, kind="active", path=str(raw), errors=errors, warnings=warnings, **extra)


def validate_artifact_path(path: str | Path, *, strict: bool = False) -> dict[str, Any]:
    """Auto-detect and validate a WaveSleuth artifact path."""
    p = Path(path)
    if p.is_dir():
        if (p / "challenge_summary.json").exists():
            return validate_challenge_directory(p, strict=strict)
        if (p / "active_summary.json").exists():
            return validate_active_directory(p)
        return _status(False, kind="directory", path=str(p), errors=["directory is not a recognized challenge or active-demo output"])
    if p.suffix == ".npz":
        return validate_run_file(p)
    if p.suffix.lower() == ".json":
        try:
            data = load_json(p)
        except Exception as exc:
            return _status(False, kind="json", path=str(p), errors=[str(exc)])
        if "grid" in data and "medium" in data and "acquisition" in data:
            return validate_world_file(p)
        if "challenge" in data and ("challenge_score" in data or "physical_score" in data or "score_summary" in data):
            return validate_challenge_directory(p, strict=strict)
        if "rounds" in data and "source_history" in data:
            return validate_active_directory(p)
        return validate_reconstruction_file(p)
    return _status(False, kind="unknown", path=str(p), errors=["unrecognized artifact type"])


def validate_artifact_paths(paths: Iterable[str | Path], *, strict: bool = False) -> dict[str, Any]:
    """Validate several artifact paths and return a combined report."""
    results = [validate_artifact_path(path, strict=strict) for path in paths]
    return {
        **base_metadata(),
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "ok": all(bool(item.get("ok")) for item in results),
        "checked": len(results),
        "results": results,
    }


def run_release_challenge_suite(
    out_dir: str | Path,
    *,
    challenges: Iterable[str] = DEFAULT_RELEASE_CHALLENGES,
    quiet: bool = False,
    blind_ellipse: bool = False,
) -> dict[str, Any]:
    """Run the standard v0.9 challenge suite and write a suite summary."""
    from .challenge import collect_leaderboard, run_challenge

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    challenge_names = [str(item) for item in challenges]
    summaries: list[dict[str, Any]] = []
    output_dirs: list[str] = []
    for name in challenge_names:
        challenge_dir = root / name
        blind = bool(blind_ellipse and name == "ellipse-easy")
        summary = run_challenge(name, out_dir=challenge_dir, quiet=quiet, clean=True, blind=blind)
        summaries.append(summary)
        output_dirs.append(str(challenge_dir))
    leaderboard = collect_leaderboard(output_dirs)
    validation = validate_artifact_paths(output_dirs)
    suite = {
        **base_metadata(),
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "suite": "v0.9-standard-challenge-suite",
        "challenge_names": challenge_names,
        "challenge_dirs": output_dirs,
        "leaderboard": leaderboard,
        "validation": validation,
        "summaries": summaries,
    }
    save_json(suite, root / "release_suite_summary.json")
    generate_release_html_report(root / "release_suite_report.html", challenge_paths=output_dirs)
    return suite


def _html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def generate_release_html_report(
    out_path: str | Path,
    *,
    challenge_paths: Iterable[str | Path] = (),
    active_paths: Iterable[str | Path] = (),
) -> Path:
    """Generate a compact HTML report for challenge and active-demo outputs."""
    from .challenge import collect_leaderboard

    out = ensure_parent(out_path)
    challenge_list = [str(p) for p in challenge_paths]
    active_list = [str(p) for p in active_paths]
    challenge_rows = collect_leaderboard(challenge_list) if challenge_list else []
    active_rows: list[dict[str, Any]] = []
    if active_list:
        try:
            from .active import collect_active_leaderboard

            active_rows = collect_active_leaderboard(active_list).get("active_leaderboard", [])
        except Exception as exc:  # pragma: no cover - depends on optional active state
            active_rows = [{"error": str(exc)}]
    validation = validate_artifact_paths([*challenge_list, *active_list]) if challenge_list or active_list else {"ok": True, "results": []}
    challenge_columns = ["challenge", "blind", "difficulty", "score", "iou", "center_error", "forward_runs", "path"]
    active_columns = ["kind", "strategy", "score", "final_reconstruction_score", "score_delta", "round_count", "path"]
    validation_rows = [
        {
            "kind": item.get("kind"),
            "ok": item.get("ok"),
            "path": item.get("path"),
            "errors": "; ".join(item.get("errors", [])),
            "warnings": "; ".join(item.get("warnings", [])),
        }
        for item in validation.get("results", [])
    ]
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>WaveSleuth-Devito v0.9 Release Report</title>
  <style>
    body {{ font-family: sans-serif; line-height: 1.4; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #f3f3f3; }}
    code {{ background: #f5f5f5; padding: 0.1rem 0.25rem; }}
  </style>
</head>
<body>
  <h1>WaveSleuth-Devito v0.9 Release Report</h1>
  <p>Package version: <code>{html.escape(__version__)}</code>. Schema version: <code>{RELEASE_SCHEMA_VERSION}</code>.</p>
  <h2>Challenge leaderboard</h2>
  {_html_table(challenge_rows, challenge_columns)}
  <h2>Active sensing leaderboard</h2>
  {_html_table(active_rows, active_columns)}
  <h2>Artifact validation</h2>
  <p>Overall validation status: <strong>{html.escape(str(validation.get('ok')))}</strong></p>
  {_html_table(validation_rows, ['kind', 'ok', 'path', 'errors', 'warnings'])}
  <h2>Notes</h2>
  <p>v0.9 is a hardening release. It does not alter Devito simulation or inversion numerics.</p>
</body>
</html>
"""
    out.write_text(html_text, encoding="utf-8")
    return out
