# WaveSleuth-Devito

WaveSleuth-Devito is a compact inverse wave playground built on Devito. It
simulates acoustic waves through hidden two dimensional media, records sparse
receiver traces, and reconstructs hidden structures with deliberately simple,
inspectable inversion strategies.

The project feels like scientific Battleship with wave propagation. We hide a
structure, choose how to illuminate it, observe a few traces, infer what is
there, and inspect why the reconstruction worked or failed.

**Status:** v1.0.0 is the first stable release. The package, fresh artifact schema, canonical challenge suite, blind challenge flow, active sensing flow, and release tooling are versioned together.

## What it is

- a local CPU playground for small two dimensional acoustic inverse problems
- a real Devito forward model, not a NumPy wave surrogate presented as Devito
- a visual sandbox for sparse acquisition, uncertainty, blind challenges, and
  active sensing
- a collection of honest baseline inversions that are easy to inspect and
  modify
- a reproducible challenge and reporting layer for comparing quality and cost

## What it is not

- a production seismic, ultrasound, or nondestructive testing package
- a validated field scale model
- a generic Devito benchmark
- a full waveform inversion framework
- evidence that mismatch derived uncertainty weights are calibrated Bayesian
  posteriors

## Current capabilities

| Area | Stable capability |
|---|---|
| Forward model | Devito backed constant density 2D acoustic wave equation |
| Sources | Ricker pulse, simultaneous or sequential shots |
| Boundaries | Reflective/default boundary or simple damping sponge, not full PML |
| Acquisition | Single, crossfire, ring, top only, left-right, limited angle |
| Data realism | Deterministic noise, dropout, amplitude jitter, and time jitter |
| Circle inversion | Joint grid search and staged center/radius/velocity search |
| Ellipse inversion | Known shape center search; broader parameter search is experimental |
| Mask inversion | Greedy coarse cell recovery for `mask-blocks` worlds |
| Objectives | Raw or differential traces, L2 or correlation, optional time gates |
| Uncertainty | Candidate and unique center mismatch-derived summaries |
| Challenges | Open and blind modes, physical and budgeted scores, leaderboards |
| Active sensing | Multi round source selection with three deterministic heuristics |
| Release tools | `doctor`, `validate`, `challenge-suite`, and `release-report` |

The detailed support table is in
[`docs/FEATURE_MATRIX.md`](docs/FEATURE_MATRIX.md).

## Installation

Python 3.10, 3.11, and 3.12 are the supported release targets.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[devito,test]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[devito,test]"
```

A lighter development install is available:

```bash
python -m pip install -e ".[test]"
```

World generation, JSON handling, scoring, validation, documentation checks, and
non Devito tests still work. Simulation and inversion fail clearly when Devito
is unavailable.

## Fastest useful start

```bash
wavesleuth-devito demo --out-dir demo_output --quiet
wavesleuth-devito challenge-suite --out-dir release_suite --quiet
wavesleuth-devito doctor --try-devito
```

Validate the canonical suite:

```bash
wavesleuth-devito validate \
  release_suite/circle-easy \
  release_suite/ellipse-easy \
  release_suite/circle-radius-velocity-staged \
  release_suite/mask-cell-easy
```

## Circle workflow

```bash
wavesleuth-devito generate-world \
  --kind circle \
  --acquisition-preset crossfire \
  --out worlds/circle.json

wavesleuth-devito simulate worlds/circle.json \
  --shot-mode sequential \
  --out runs/circle_obs.npz \
  --quiet

wavesleuth-devito invert runs/circle_obs.npz \
  --method grid-search \
  --mismatch-mode differential \
  --candidate-grid-size 5 \
  --refine-levels 1 \
  --out runs/circle_recon.json \
  --quiet

wavesleuth-devito visualize-reconstruction runs/circle_recon.json \
  --out figures/circle_recon.png

wavesleuth-devito score worlds/circle.json runs/circle_recon.json
```

For unknown center, radius, and anomaly velocity, use staged search:

```bash
wavesleuth-devito invert runs/circle_obs.npz \
  --method staged-grid-search \
  --search-radius \
  --search-velocity \
  --top-k-refine 5 \
  --final-refine-top-k 1 \
  --out runs/circle_staged_recon.json \
  --quiet
```

## Ellipse workflow

The stable ellipse baseline assumes that axes, orientation, and anomaly
velocity are public and searches the center.

```bash
wavesleuth-devito generate-world \
  --kind ellipse \
  --acquisition-preset crossfire \
  --out worlds/ellipse.json

wavesleuth-devito simulate worlds/ellipse.json \
  --shot-mode sequential \
  --out runs/ellipse_obs.npz \
  --quiet

wavesleuth-devito invert runs/ellipse_obs.npz \
  --method ellipse-grid-search \
  --candidate-grid-size 5 \
  --refine-levels 1 \
  --out runs/ellipse_recon.json \
  --quiet
```

`--search-axes`, `--search-angle`, and `--search-velocity` remain experimental.
Extra unknowns can reduce trace mismatch without improving geometry.

## Coarse mask workflow

`mask-blocks` is the first non parametric baseline. It greedily adds coarse
cells and verifies every candidate with Devito.

```bash
wavesleuth-devito generate-world \
  --kind mask-blocks \
  --acquisition-preset crossfire \
  --out worlds/mask_blocks.json

wavesleuth-devito simulate worlds/mask_blocks.json \
  --shot-mode sequential \
  --out runs/mask_blocks_obs.npz \
  --quiet

wavesleuth-devito invert runs/mask_blocks_obs.npz \
  --method cell-search \
  --cell-grid-size 6 \
  --max-active-cells 5 \
  --mismatch-mode differential \
  --out runs/mask_blocks_recon.json \
  --quiet
```

The bundled `mask-cell-easy` challenge is a calibrated deterministic proof of
concept, not evidence that greedy cell search is a general imaging method.

## Challenges and blind mode

```bash
wavesleuth-devito challenge circle-easy --out-dir challenge_circle --quiet
wavesleuth-devito challenge ellipse-easy --out-dir challenge_ellipse --quiet
wavesleuth-devito leaderboard challenge_circle challenge_ellipse
```

Blind mode separates public observations from the local answer key:

```bash
wavesleuth-devito challenge ellipse-easy \
  --blind \
  --out-dir challenge_ellipse_blind \
  --quiet

wavesleuth-devito score-challenge challenge_ellipse_blind
```

Public blind runs retain acquisition geometry, traces, timing, background
velocity, and known shape hints required by the stable baseline. They do not
contain the true velocity model, final wavefield, snapshots, or hidden location.

## Active sensing

```bash
wavesleuth-devito active-demo \
  --strategy uncertainty \
  --rounds 3 \
  --out-dir active_uncertainty \
  --quiet

wavesleuth-devito active-demo \
  --strategy spread \
  --rounds 3 \
  --out-dir active_spread \
  --quiet

wavesleuth-devito active-demo \
  --strategy opposite-best \
  --rounds 3 \
  --out-dir active_opposite \
  --quiet

wavesleuth-devito active-leaderboard \
  active_uncertainty active_spread active_opposite
```

These policies are deterministic heuristics, not optimal experimental design
solvers. A new shot may expose ambiguity without monotonically improving a
simple inversion.

## Scoring

Two scores serve different purposes:

- `physical_score` describes reconstruction quality, including IoU, center
  error, shape-parameter errors, and contrast errors where supported.
- `challenge_score` is the budgeted game score. It rewards reconstruction
  quality and penalizes source, receiver, and forward run cost. Runtime is
  diagnostic only.

The legacy `score` alias remains readable for backward compatibility.

## Uncertainty

Candidate probabilities are soft weights derived from mismatch values. They are
not calibrated posteriors. `center_effective_candidates` is usually the clearest
location ambiguity diagnostic because it deduplicates repeated centers across
refinement stages.

## World families

Generation and forward simulation support:

```text
circle
rectangle
layered
blobs
ellipse
ring
two-circles
crack
circle-layered
mask-blocks
```

Stable reconstruction baselines currently cover circle, known shape ellipse,
and calibrated coarse `mask-blocks` targets. Other families are simulation and
visualization targets for future inversion work.

## Validation and release audit

```bash
python -m pytest
wavesleuth-devito doctor --try-devito
wavesleuth-devito self-test --try-devito
wavesleuth-devito challenge-suite --out-dir release_suite --quiet
bash collect_wavesleuth_v1_audit.sh
```

The master audit collects tests, environment information, a world gallery, the
standard challenge suite, blind integrity checks, active demos, validation,
reports, and checksums.

## Current limitations

- The forward model is a small constant density acoustic toy model.
- The simple sponge is not a production PML.
- Circle and ellipse inversions are parametric.
- Ellipse axes/orientation recovery is experimental.
- Mask cell search is greedy and calibrated only for a simple deterministic
  milestone.
- Velocity contrast remains less identifiable than location and radius in the
  staged circle challenge.
- Active policies are heuristic.
- Uncertainty weights are qualitative.
- No GPU, MPI, 3D, cloud, notebook, or neural network dependency is required.

## Documentation

- [`examples/quickstart.md`](examples/quickstart.md)
- [`docs/SCHEMA.md`](docs/SCHEMA.md)
- [`docs/FEATURE_MATRIX.md`](docs/FEATURE_MATRIX.md)
- [`docs/V1_RELEASE_CHECKLIST.md`](docs/V1_RELEASE_CHECKLIST.md)
- [`docs/POST_V1_ROADMAP.md`](docs/POST_V1_ROADMAP.md)
- [`CHANGELOG.md`](CHANGELOG.md)

## License

MIT.
## Release validation

```bash
python tools/release_gate.py --with-devito
bash collect_wavesleuth_v1_audit.sh
```

The release gate runs compilation, pytest, dependency and health checks, plus an
isolated wheel build/install/CLI smoke test.

## v1 release and future work

See `docs/V1_RELEASE_NOTES.md` and `docs/POST_V1_ROADMAP.md`.
