"""Command-line interface for WaveSleuth-Devito."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .exceptions import DevitoUnavailableError, WaveSleuthError
from .examples import run_demo
from .inversion import grid_search_circle
from .io import load_json, load_world, save_json, save_world
from .metadata import PROJECT_NAME, __version__
from .scoring import score_reconstruction
from .simulation import simulate_world
from .visualization import visualize_reconstruction, visualize_run, visualize_world
from .world import SUPPORTED_WORLD_KINDS, make_default_world, make_demo_world, velocity_model_from_world


def _json_print(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


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
    p_generate.set_defaults(func=cmd_generate_world)

    p_simulate = subparsers.add_parser("simulate", help="Run Devito acoustic simulation for a world.")
    p_simulate.add_argument("world", help="Input world JSON path.")
    p_simulate.add_argument("--out", required=True, help="Output run .npz path.")
    p_simulate.add_argument("--quiet", action="store_true", help="Reduce Devito logging.")
    p_simulate.add_argument("--no-wavefield", action="store_true", help="Do not save final wavefield/snapshots.")
    p_simulate.set_defaults(func=cmd_simulate)

    p_invert = subparsers.add_parser("invert", help="Invert observed traces with a simple search method.")
    p_invert.add_argument("run", help="Input observed run .npz path.")
    p_invert.add_argument("--method", choices=["grid-search"], default="grid-search")
    p_invert.add_argument("--out", required=True, help="Output reconstruction JSON path.")
    p_invert.add_argument("--candidate-grid-size", type=int, default=5)
    p_invert.add_argument("--radius", type=float, default=None)
    p_invert.add_argument("--anomaly-velocity", type=float, default=None)
    p_invert.add_argument("--max-candidates", type=int, default=None)
    p_invert.add_argument("--quiet", action="store_true")
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

    p_score = subparsers.add_parser("score", help="Score a reconstruction against a true world.")
    p_score.add_argument("world", help="True world JSON path.")
    p_score.add_argument("reconstruction", help="Reconstruction JSON path.")
    p_score.set_defaults(func=cmd_score)

    p_demo = subparsers.add_parser("demo", help="Run a complete tiny pipeline.")
    p_demo.add_argument("--out-dir", required=True, help="Output directory for demo files.")
    p_demo.add_argument("--candidate-grid-size", type=int, default=5)
    p_demo.add_argument("--quiet", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_self = subparsers.add_parser("self-test", help="Run lightweight sanity checks.")
    p_self.add_argument("--try-devito", action="store_true", help="Run a tiny Devito simulation when Devito is installed.")
    p_self.set_defaults(func=cmd_self_test)

    return parser


def cmd_generate_world(args: argparse.Namespace) -> int:
    world = make_default_world(args.kind, seed=args.seed, name=args.name)
    save_world(world, args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    world = load_world(args.world)
    simulate_world(world, out_path=args.out, save_wavefield=not args.no_wavefield, quiet=args.quiet)
    print(f"wrote {args.out}")
    return 0


def cmd_invert(args: argparse.Namespace) -> int:
    if args.method != "grid-search":
        raise WaveSleuthError(f"Unsupported inversion method {args.method!r}")
    reconstruction = grid_search_circle(
        args.run,
        out_path=args.out,
        candidate_grid_size=args.candidate_grid_size,
        radius=args.radius,
        anomaly_velocity=args.anomaly_velocity,
        max_candidates=args.max_candidates,
        quiet=args.quiet,
    )
    print(f"wrote {args.out}")
    _json_print({"best_candidate": reconstruction["best_candidate"], "score": reconstruction.get("score")})
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


def cmd_score(args: argparse.Namespace) -> int:
    world = load_world(args.world)
    reconstruction = load_json(args.reconstruction)
    _json_print(score_reconstruction(world, reconstruction))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    summary = run_demo(args.out_dir, candidate_grid_size=args.candidate_grid_size, quiet=args.quiet)
    _json_print(summary)
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    circle = make_default_world("circle")
    model = velocity_model_from_world(circle)
    if model.shape != (circle["grid"]["nx"], circle["grid"]["nz"]):
        raise WaveSleuthError("velocity model shape check failed")

    blob_a = make_default_world("blobs", seed=123)
    blob_b = make_default_world("blobs", seed=123)
    if blob_a["medium"]["anomaly"] != blob_b["medium"]["anomaly"]:
        raise WaveSleuthError("deterministic blob generation check failed")

    messages = ["world generation: ok", "velocity model: ok", "deterministic blobs: ok"]
    if args.try_devito:
        try:
            tiny = make_demo_world()
            tiny["grid"].update({"nx": 24, "nz": 24, "extent_x": 0.35, "extent_z": 0.35})
            tiny["medium"]["anomaly"].update({"center_x": 0.18, "center_z": 0.18, "radius": 0.045})
            tiny["acquisition"]["sources"] = [{"x": 0.10, "z": 0.10}]
            tiny["acquisition"]["receivers"] = [{"x": 0.15, "z": 0.23}, {"x": 0.22, "z": 0.23}]
            tiny["simulation"].update({"nt": 50, "dt": 0.001, "source_frequency": 25.0})
            result = simulate_world(tiny, save_wavefield=False, quiet=True)
            messages.append(f"tiny Devito simulation: ok, traces={result.receiver_traces.shape}")
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
