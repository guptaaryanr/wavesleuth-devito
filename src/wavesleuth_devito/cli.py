"""Command-line interface for WaveSleuth-Devito."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .challenge import SUPPORTED_CHALLENGES, collect_leaderboard, run_challenge
from .exceptions import DevitoUnavailableError, WaveSleuthError
from .examples import run_demo
from .experiments import compare_acquisitions
from .inversion import grid_search_circle
from .io import load_json, load_world, save_world
from .metadata import PROJECT_NAME, __version__
from .report import generate_html_report
from .scoring import score_reconstruction
from .simulation import apply_trace_noise, simulate_world, sponge_damping_model
from .visualization import visualize_reconstruction, visualize_run, visualize_uncertainty, visualize_world
from .world import (
    SUPPORTED_ACQUISITION_PRESETS,
    SUPPORTED_BOUNDARIES,
    SUPPORTED_WORLD_KINDS,
    make_default_world,
    make_demo_world,
    velocity_model_from_world,
)


def _json_print(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_float_csv(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values: list[float] = []
    for item in _parse_csv(text):
        try:
            values.append(float(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Could not parse float value {item!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wavesleuth-devito",
        description="Scientific Battleship with Devito acoustic waves.",
    )
    parser.add_argument("--version", action="version", version=f"{PROJECT_NAME} {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_generate = subparsers.add_parser("generate-world", help="Generate a deterministic hidden world JSON file.")
    p_generate.add_argument("--kind", choices=SUPPORTED_WORLD_KINDS, required=True)
    p_generate.add_argument("--out", required=True, help="Output world JSON path.")
    p_generate.add_argument("--seed", type=int, default=20260203, help="Seed used by random world types.")
    p_generate.add_argument("--name", default=None, help="Optional world name.")
    p_generate.add_argument("--acquisition-preset", choices=SUPPORTED_ACQUISITION_PRESETS, default="single")
    p_generate.add_argument("--boundary", choices=SUPPORTED_BOUNDARIES, default=None, help="Optional boundary mode metadata.")
    p_generate.add_argument("--sponge-width", type=int, default=None)
    p_generate.add_argument("--sponge-strength", type=float, default=None)
    p_generate.set_defaults(func=cmd_generate_world)

    p_simulate = subparsers.add_parser("simulate", help="Run Devito acoustic simulation for a world.")
    p_simulate.add_argument("world", help="Input world JSON path.")
    p_simulate.add_argument("--out", required=True, help="Output run .npz path.")
    p_simulate.add_argument("--quiet", action="store_true", help="Reduce Devito logging.")
    p_simulate.add_argument("--no-wavefield", action="store_true", help="Do not save final wavefield/snapshots.")
    p_simulate.add_argument("--shot-mode", choices=["simultaneous", "sequential"], default=None)
    p_simulate.add_argument("--boundary", choices=SUPPORTED_BOUNDARIES, default=None)
    p_simulate.add_argument("--sponge-width", type=int, default=None)
    p_simulate.add_argument("--sponge-strength", type=float, default=None)
    p_simulate.add_argument("--noise-level", type=float, default=None, help="Add Gaussian noise relative to trace RMS.")
    p_simulate.add_argument("--receiver-dropout", type=float, default=None, help="Randomly zero a fraction of receiver channels.")
    p_simulate.add_argument("--amplitude-jitter", type=float, default=None, help="Per-channel amplitude gain jitter.")
    p_simulate.add_argument("--time-jitter", type=float, default=None, help="Per-channel timing jitter in seconds.")
    p_simulate.add_argument("--noise-seed", type=int, default=None)
    p_simulate.set_defaults(func=cmd_simulate)

    p_invert = subparsers.add_parser("invert", help="Invert observed traces with a simple search method.")
    p_invert.add_argument("run", help="Input observed run .npz path.")
    p_invert.add_argument("--method", choices=["grid-search", "staged-grid-search"], default="grid-search", help="Use grid-search with optional search strategy; staged-grid-search is an alias for --search-strategy staged.")
    p_invert.add_argument("--out", required=True, help="Output reconstruction JSON path.")
    p_invert.add_argument("--candidate-grid-size", type=int, default=5)
    p_invert.add_argument("--radius", type=float, default=None)
    p_invert.add_argument("--anomaly-velocity", type=float, default=None)
    p_invert.add_argument("--radius-values", type=_parse_float_csv, default=None, help="Comma-separated radius values, e.g. 0.09,0.12,0.15")
    p_invert.add_argument("--anomaly-velocity-values", type=_parse_float_csv, default=None, help="Comma-separated anomaly velocities.")
    p_invert.add_argument("--search-radius", action="store_true", help="Search a tiny default radius axis around the metadata radius.")
    p_invert.add_argument("--search-velocity", action="store_true", help="Search a tiny default velocity axis around the metadata velocity.")
    p_invert.add_argument("--max-candidates", type=int, default=None)
    p_invert.add_argument("--quiet", action="store_true")
    p_invert.add_argument("--mismatch-mode", choices=["raw", "differential"], default="differential")
    p_invert.add_argument("--metric", choices=["l2", "correlation"], default="l2")
    p_invert.add_argument("--time-min", type=float, default=None)
    p_invert.add_argument("--time-max", type=float, default=None)
    p_invert.add_argument("--normalize-traces", action="store_true")
    p_invert.add_argument("--refine-levels", type=int, default=0)
    p_invert.add_argument("--shot-mode", choices=["simultaneous", "sequential"], default=None)
    p_invert.add_argument("--search-strategy", choices=["auto", "joint", "staged"], default="auto", help="v0.4 strategy: auto uses staged when radius/velocity axes are searched.")
    p_invert.add_argument("--top-k-refine", type=int, default=5, help="For staged search, keep this many center candidates for refinement.")
    p_invert.add_argument("--final-refine-top-k", type=int, default=1, help="For staged search, final center polish around this many parameter candidates.")
    p_invert.add_argument("--center-metric", choices=["l2", "correlation"], default=None, help="Metric for staged center screening. Defaults to --metric.")
    p_invert.add_argument("--final-metric", choices=["l2", "correlation"], default=None, help="Metric for final staged center polish. Defaults to --metric.")
    p_invert.add_argument("--parameter-prior", choices=["none", "reference"], default="none", help="Optional weak reference prior for radius/velocity search.")
    p_invert.add_argument("--radius-prior-weight", type=float, default=0.0)
    p_invert.add_argument("--velocity-prior-weight", type=float, default=0.0)
    p_invert.set_defaults(func=cmd_invert)

    p_vworld = subparsers.add_parser("visualize-world", help="Plot a world velocity model.")
    p_vworld.add_argument("world", help="Input world JSON path.")
    p_vworld.add_argument("--out", required=True, help="Output PNG path.")
    p_vworld.set_defaults(func=cmd_visualize_world)

    p_vrun = subparsers.add_parser("visualize-run", help="Plot receiver traces from a run.")
    p_vrun.add_argument("run", help="Input run .npz path.")
    p_vrun.add_argument("--out", required=True, help="Output PNG path.")
    p_vrun.set_defaults(func=cmd_visualize_run)

    p_vrecon = subparsers.add_parser("visualize-reconstruction", help="Plot reconstruction summary.")
    p_vrecon.add_argument("reconstruction", help="Input reconstruction JSON path.")
    p_vrecon.add_argument("--out", required=True, help="Output PNG path.")
    p_vrecon.set_defaults(func=cmd_visualize_reconstruction)

    p_vunc = subparsers.add_parser("visualize-uncertainty", help="Plot uncertainty/pseudo-probability from a reconstruction mismatch map.")
    p_vunc.add_argument("reconstruction", help="Input reconstruction JSON path.")
    p_vunc.add_argument("--out", required=True, help="Output PNG path.")
    p_vunc.add_argument("--temperature", type=float, default=None)
    p_vunc.set_defaults(func=cmd_visualize_uncertainty)

    p_score = subparsers.add_parser("score", help="Score a reconstruction against a true world.")
    p_score.add_argument("world", help="True world JSON path.")
    p_score.add_argument("reconstruction", help="Reconstruction JSON path.")
    p_score.set_defaults(func=cmd_score)

    p_report = subparsers.add_parser("report", help="Generate a lightweight HTML experiment report.")
    p_report.add_argument("reconstruction", help="Input reconstruction JSON path.")
    p_report.add_argument("--out", required=True, help="Output HTML path.")
    p_report.set_defaults(func=cmd_report)

    p_compare = subparsers.add_parser("compare-acquisition", help="Compare acquisition presets on the same hidden world.")
    p_compare.add_argument("world", help="Input base world JSON path.")
    p_compare.add_argument("--out-dir", required=True)
    p_compare.add_argument("--presets", default="single,crossfire,ring,top-only,left-right")
    p_compare.add_argument("--candidate-grid-size", type=int, default=5)
    p_compare.add_argument("--refine-levels", type=int, default=0)
    p_compare.add_argument("--mismatch-mode", choices=["raw", "differential"], default="differential")
    p_compare.add_argument("--metric", choices=["l2", "correlation"], default="l2")
    p_compare.add_argument("--quiet", action="store_true")
    p_compare.set_defaults(func=cmd_compare_acquisition)

    p_challenge = subparsers.add_parser("challenge", help="Run a named budgeted challenge.")
    p_challenge.add_argument("name", choices=SUPPORTED_CHALLENGES)
    p_challenge.add_argument("--out-dir", required=True)
    p_challenge.add_argument("--candidate-grid-size", type=int, default=None)
    p_challenge.add_argument("--refine-levels", type=int, default=None)
    p_challenge.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        default=True,
        help="Clean challenge-owned outputs before running. This is the default in v0.3.2.",
    )
    p_challenge.add_argument(
        "--no-clean",
        "--keep-existing",
        dest="clean",
        action="store_false",
        help="Preserve existing files in the challenge output directory.",
    )
    p_challenge.add_argument("--quiet", action="store_true")
    p_challenge.set_defaults(func=cmd_challenge)

    p_leader = subparsers.add_parser("leaderboard", help="Collect challenge_summary.json files into a sorted leaderboard.")
    p_leader.add_argument("paths", nargs="+", help="Files or directories to scan.")
    p_leader.set_defaults(func=cmd_leaderboard)

    p_demo = subparsers.add_parser("demo", help="Run a complete tiny pipeline.")
    p_demo.add_argument("--out-dir", required=True, help="Output directory for demo files.")
    p_demo.add_argument("--candidate-grid-size", type=int, default=5)
    p_demo.add_argument("--refine-levels", type=int, default=1)
    p_demo.add_argument("--mismatch-mode", choices=["raw", "differential"], default="differential")
    p_demo.add_argument("--metric", choices=["l2", "correlation"], default="l2")
    p_demo.add_argument("--search-strategy", choices=["auto", "joint", "staged"], default="auto")
    p_demo.add_argument("--top-k-refine", type=int, default=5)
    p_demo.add_argument("--final-refine-top-k", type=int, default=1)
    p_demo.add_argument("--search-radius", action="store_true")
    p_demo.add_argument("--search-velocity", action="store_true")
    p_demo.add_argument("--noise-level", type=float, default=0.0)
    p_demo.add_argument("--quiet", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_self = subparsers.add_parser("self-test", help="Run lightweight sanity checks.")
    p_self.add_argument("--try-devito", action="store_true", help="Run a tiny Devito simulation when Devito is installed.")
    p_self.set_defaults(func=cmd_self_test)

    return parser


def _apply_simulation_overrides(world: dict[str, Any], args: argparse.Namespace) -> None:
    if getattr(args, "boundary", None) is not None:
        world["simulation"]["boundary"] = args.boundary
    if getattr(args, "sponge_width", None) is not None:
        world["simulation"]["sponge_width"] = int(args.sponge_width)
    if getattr(args, "sponge_strength", None) is not None:
        world["simulation"]["sponge_strength"] = float(args.sponge_strength)


def cmd_generate_world(args: argparse.Namespace) -> int:
    world = make_default_world(args.kind, seed=args.seed, name=args.name, acquisition=args.acquisition_preset)
    _apply_simulation_overrides(world, args)
    save_world(world, args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    world = load_world(args.world)
    _apply_simulation_overrides(world, args)
    simulate_world(
        world,
        out_path=args.out,
        save_wavefield=not args.no_wavefield,
        quiet=args.quiet,
        shot_mode=args.shot_mode,
        noise_level=args.noise_level,
        receiver_dropout=args.receiver_dropout,
        amplitude_jitter=args.amplitude_jitter,
        time_jitter=args.time_jitter,
        noise_seed=args.noise_seed,
    )
    print(f"wrote {args.out}")
    return 0


def cmd_invert(args: argparse.Namespace) -> int:
    search_strategy = args.search_strategy
    if args.method == "staged-grid-search":
        search_strategy = "staged"
    elif args.method != "grid-search":
        raise WaveSleuthError(f"Unsupported inversion method {args.method!r}")
    reconstruction = grid_search_circle(
        args.run,
        out_path=args.out,
        candidate_grid_size=args.candidate_grid_size,
        radius=args.radius,
        anomaly_velocity=args.anomaly_velocity,
        radius_values=args.radius_values,
        anomaly_velocity_values=args.anomaly_velocity_values,
        search_radius=args.search_radius,
        search_velocity=args.search_velocity,
        max_candidates=args.max_candidates,
        quiet=args.quiet,
        mismatch_mode=args.mismatch_mode,
        metric=args.metric,
        time_min=args.time_min,
        time_max=args.time_max,
        normalize_traces=args.normalize_traces,
        refine_levels=args.refine_levels,
        shot_mode=args.shot_mode,
        search_strategy=search_strategy,
        top_k_refine=args.top_k_refine,
        final_refine_top_k=args.final_refine_top_k,
        center_metric=args.center_metric,
        final_metric=args.final_metric,
        parameter_prior=args.parameter_prior,
        radius_prior_weight=args.radius_prior_weight,
        velocity_prior_weight=args.velocity_prior_weight,
    )
    print(f"wrote {args.out}")
    _json_print(
        {
            "objective": reconstruction.get("objective"),
            "search": reconstruction.get("search"),
            "best_candidate": reconstruction["best_candidate"],
            "nearest_true_candidate": reconstruction.get("nearest_true_candidate"),
            "score": reconstruction.get("score"),
            "candidate_grid": reconstruction.get("candidate_grid"),
        }
    )
    return 0


def cmd_visualize_world(args: argparse.Namespace) -> int:
    out = visualize_world(args.world, args.out)
    print(f"wrote {out}")
    return 0


def cmd_visualize_run(args: argparse.Namespace) -> int:
    out = visualize_run(args.run, args.out)
    print(f"wrote {out}")
    return 0


def cmd_visualize_reconstruction(args: argparse.Namespace) -> int:
    out = visualize_reconstruction(args.reconstruction, args.out)
    print(f"wrote {out}")
    return 0


def cmd_visualize_uncertainty(args: argparse.Namespace) -> int:
    out = visualize_uncertainty(args.reconstruction, args.out, temperature=args.temperature)
    print(f"wrote {out}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    world = load_world(args.world)
    reconstruction = load_json(args.reconstruction)
    _json_print(score_reconstruction(world, reconstruction))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    out = generate_html_report(args.reconstruction, args.out)
    print(f"wrote {out}")
    return 0


def cmd_compare_acquisition(args: argparse.Namespace) -> int:
    presets = _parse_csv(args.presets)
    summary = compare_acquisitions(
        args.world,
        out_dir=args.out_dir,
        presets=presets,
        candidate_grid_size=args.candidate_grid_size,
        refine_levels=args.refine_levels,
        mismatch_mode=args.mismatch_mode,
        metric=args.metric,
        quiet=args.quiet,
    )
    _json_print(summary)
    return 0


def cmd_challenge(args: argparse.Namespace) -> int:
    summary = run_challenge(
        args.name,
        out_dir=args.out_dir,
        candidate_grid_size=args.candidate_grid_size,
        refine_levels=args.refine_levels,
        quiet=args.quiet,
        clean=args.clean,
    )
    _json_print(summary)
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    _json_print({"leaderboard": collect_leaderboard(args.paths)})
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    summary = run_demo(
        args.out_dir,
        candidate_grid_size=args.candidate_grid_size,
        refine_levels=args.refine_levels,
        mismatch_mode=args.mismatch_mode,
        metric=args.metric,
        search_strategy=args.search_strategy,
        top_k_refine=args.top_k_refine,
        final_refine_top_k=args.final_refine_top_k,
        search_radius=args.search_radius,
        search_velocity=args.search_velocity,
        noise_level=args.noise_level,
        quiet=args.quiet,
    )
    _json_print(summary)
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    circle = make_default_world("circle")
    model = velocity_model_from_world(circle)
    if model.shape != (circle["grid"]["nx"], circle["grid"]["nz"]):
        raise WaveSleuthError("velocity model shape check failed")

    ring = make_default_world("circle", acquisition="ring")
    if len(ring["acquisition"]["sources"]) < 4 or len(ring["acquisition"]["receivers"]) < 8:
        raise WaveSleuthError("ring acquisition check failed")

    noisy = apply_trace_noise(model[:10, :3], dt=0.001, noise_level=0.01, seed=123)
    if noisy.shape != model[:10, :3].shape:
        raise WaveSleuthError("noise helper shape check failed")

    demo = make_demo_world()
    damp = sponge_damping_model(demo)
    if damp.shape != (demo["grid"]["nx"], demo["grid"]["nz"]):
        raise WaveSleuthError("sponge damping shape check failed")

    messages = [
        "world generation: ok",
        "velocity model: ok",
        "ring acquisition: ok",
        "noise helper: ok",
        "sponge damping helper: ok",
    ]
    if args.try_devito:
        try:
            tiny = make_demo_world()
            tiny["grid"].update({"nx": 24, "nz": 24, "extent_x": 0.35, "extent_z": 0.35})
            tiny["medium"]["anomaly"].update({"center_x": 0.18, "center_z": 0.18, "radius": 0.045})
            tiny["acquisition"]["sources"] = [{"x": 0.10, "z": 0.10}, {"x": 0.25, "z": 0.10}]
            tiny["acquisition"]["receivers"] = [{"x": 0.15, "z": 0.23}, {"x": 0.22, "z": 0.23}]
            tiny["simulation"].update({"nt": 50, "dt": 0.001, "source_frequency": 25.0, "shot_mode": "sequential", "sponge_width": 3})
            result = simulate_world(tiny, save_wavefield=False, quiet=True, shot_mode="sequential", noise_level=0.001)
            messages.append(f"tiny Devito sequential simulation: ok, traces={result.receiver_traces.shape}")
        except DevitoUnavailableError:
            messages.append("tiny Devito simulation: skipped, Devito is not installed")
    else:
        messages.append("tiny Devito simulation: not requested; pass --try-devito to run it")

    for message in messages:
        print(message)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WaveSleuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
