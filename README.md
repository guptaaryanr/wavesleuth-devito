# WaveSleuth-Devito

WaveSleuth-Devito is a toy inverse-physics playground I made just for fun and because I was bored and wanted to try something new.

It uses Devito to simulate acoustic waves through hidden 2D media, records sparse receiver traces, and tries to reconstruct hidden structures using simple inversion strategies.

The vibe is scientific Battleship with wave propagation: hide something in a medium, fire waves through it, observe only a few traces, then make a guess about what was hidden.

## What this is

WaveSleuth-Devito is a small, runnable sandbox for learning and tinkering with:

- wave propagation
- hidden-medium reconstruction
- inverse problems
- sparse sensing
- source and receiver placement
- simple search-based inversion
- visualization
- future AI-for-science extensions

## What this is not

This is not a paper, not a production inversion package, not a generic Devito benchmark, and not a polished full-waveform inversion framework. The MVP intentionally favors a hackable end-to-end pipeline over architectural depth.

## Why it exists

A lot of inverse-problem software jumps from textbook math to heavyweight research systems. WaveSleuth-Devito tries to make the basic loop tangible:

1. define a hidden 2D world
2. simulate acoustic waves through it
3. record sparse receiver traces
4. search over candidate hidden structures
5. visualize and score the guess

The default inversion is deliberately simple: a coarse grid search for the center of a circular anomaly. It is not smart, but it is inspectable.

## Installation

For the full wave simulation path, install with the Devito extra:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[devito,test]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[devito,test]"
```

A lighter install works for JSON generation, scoring helpers, and non-Devito tests:

```bash
python -m pip install -e ".[test]"
```

Simulation and inversion commands will clearly fail if Devito is not installed.

## Quickstart

```bash
wavesleuth-devito generate-world --kind circle --out worlds/circle.json
wavesleuth-devito simulate worlds/circle.json --out runs/circle_obs.npz
wavesleuth-devito invert runs/circle_obs.npz --method grid-search --out runs/circle_recon.json
wavesleuth-devito visualize-world worlds/circle.json --out figures/circle_world.png
wavesleuth-devito visualize-run runs/circle_obs.npz --out figures/circle_traces.png
wavesleuth-devito visualize-reconstruction runs/circle_recon.json --out figures/circle_recon.png
wavesleuth-devito score worlds/circle.json runs/circle_recon.json
```

Or run the whole tiny pipeline:

```bash
wavesleuth-devito demo --out-dir demo_output
```

## CLI commands

```bash
wavesleuth-devito --help

wavesleuth-devito generate-world --kind circle --out worlds/circle.json
wavesleuth-devito generate-world --kind rectangle --out worlds/rectangle.json
wavesleuth-devito generate-world --kind layered --out worlds/layered.json
wavesleuth-devito generate-world --kind blobs --out worlds/blobs.json

wavesleuth-devito simulate worlds/circle.json --out runs/circle_obs.npz
wavesleuth-devito invert runs/circle_obs.npz --method grid-search --out runs/circle_recon.json
wavesleuth-devito visualize-world worlds/circle.json --out figures/circle_world.png
wavesleuth-devito visualize-run runs/circle_obs.npz --out figures/circle_traces.png
wavesleuth-devito visualize-reconstruction runs/circle_recon.json --out figures/circle_recon.png

wavesleuth-devito score worlds/circle.json runs/circle_recon.json
wavesleuth-devito demo --out-dir demo_output
wavesleuth-devito self-test
```

## Worlds

A world is a JSON file containing grid geometry, velocity settings, hidden-medium parameters, source and receiver coordinates, and simulation settings.

Supported generated worlds:

- `circle`: one circular velocity anomaly
- `rectangle`: one rectangular velocity anomaly
- `layered`: horizontal velocity layers
- `blobs`: multiple deterministic random circular anomalies

Example:

```json
{
  "name": "circle_demo",
  "grid": {
    "nx": 70,
    "nz": 70,
    "extent_x": 1.0,
    "extent_z": 1.0
  },
  "medium": {
    "background_velocity": 1.5,
    "anomaly_velocity": 2.2,
    "anomaly": {
      "kind": "circle",
      "center_x": 0.55,
      "center_z": 0.52,
      "radius": 0.12
    }
  },
  "acquisition": {
    "sources": [
      {"x": 0.2, "z": 0.12}
    ],
    "receivers": [
      {"x": 0.15, "z": 0.82},
      {"x": 0.30, "z": 0.82},
      {"x": 0.45, "z": 0.82},
      {"x": 0.60, "z": 0.82},
      {"x": 0.75, "z": 0.82},
      {"x": 0.90, "z": 0.82}
    ]
  },
  "simulation": {
    "nt": 360,
    "dt": 0.0015,
    "space_order": 4,
    "source_frequency": 20.0
  }
}
```

Coordinates are physical coordinates in the domain described by `extent_x` and `extent_z`.

## Simulation

The forward simulation uses a simple constant-density acoustic wave equation in 2D, implemented with Devito symbols and operators. Sources and receivers are sparse Devito time functions. The source pulse is a Ricker wavelet.

The generated `.npz` run file contains:

- `receiver_traces`: array with shape `(nt, n_receivers)`
- `time`: simulation times
- `velocity_model`: 2D velocity grid
- `source_coordinates`: source locations as `(x, z)`
- `receiver_coordinates`: receiver locations as `(x, z)`
- `final_wavefield`: final wavefield if saved
- `snapshots`: sparse wavefield snapshots if saved
- `world_json`: serialized world metadata

MVP boundary behavior is intentionally crude. The solver does not implement a production absorbing boundary or PML. Expect boundary reflections, especially for long runs or sources near the edges.

## Inversion

The first inversion method is `grid-search` for circular anomalies.

Given observed traces from a circle world, the inversion:

1. reads the hidden-world metadata stored inside the `.npz`
2. holds radius and anomaly velocity fixed unless overridden
3. scans a coarse grid of candidate circle centers
4. runs a Devito forward simulation for each candidate
5. computes normalized trace mismatch
6. saves the best candidate and mismatch map

Example:

```bash
wavesleuth-devito invert runs/circle_obs.npz \
  --method grid-search \
  --candidate-grid-size 5 \
  --out runs/circle_recon.json
```

Useful options:

```bash
--candidate-grid-size 7
--radius 0.10
--anomaly-velocity 2.1
--max-candidates 20
--quiet
```

This is intentionally brute force. It is there to make the inverse loop visible, not to be clever.

## Scoring

For circle worlds, scoring reports:

- center error
- normalized center error
- radius error
- anomaly mask IoU
- best mismatch if available

For non-circle worlds, the MVP scorer returns a clear unsupported message instead of pretending to evaluate a method it does not yet understand.

## Visualization

Matplotlib visualizations are available for:

- worlds: velocity model plus source and receiver overlays
- runs: receiver trace heatmap
- reconstructions: true/predicted circles plus candidate mismatch heatmap

No notebooks are required.

## Current limitations

- Circle inversion only searches for the anomaly center.
- Radius and anomaly velocity are fixed or supplied manually.
- The default acoustic solver uses simple zero-style boundary behavior, not a tuned absorbing boundary.
- One or more sources are supported, but the default examples use a single simultaneous shot.
- The inversion repeatedly runs forward models, so it is intentionally small.
- The numerical model is for learning and play, not validated field-scale modeling.

## Future ideas

- better absorbing boundaries
- multiple shot experiments
- active source/receiver placement
- CO2 plume toy mode
- ultrasound mode
- nondestructive testing mode
- learned inversion using small neural networks
- surrogate forward models
- uncertainty maps
- comparison against stronger inversion methods
- performance and validation reports as secondary support features

## Development checks

```bash
python -m pytest
wavesleuth-devito self-test
```

Devito-heavy tests are skipped when Devito is unavailable.
## v0.2 improvements

This repo has been upgraded with a more useful inverse-problem baseline:

- `--acquisition-preset crossfire` generates sparse multi-angle source/receiver geometry.
- `simulate --shot-mode sequential` fires sources one at a time and stores a trace cube.
- `invert --mismatch-mode differential` subtracts a background simulation before comparing traces.
- `invert --refine-levels N` performs local grid refinement after the first coarse search.
- `invert --metric correlation`, `--time-min`, `--time-max`, and `--normalize-traces` expose a few simple objective variants.

A good v0.2 workflow is:

```bash
wavesleuth-devito generate-world --kind circle --acquisition-preset crossfire --out worlds/circle_crossfire.json
wavesleuth-devito simulate worlds/circle_crossfire.json --shot-mode sequential --out runs/circle_crossfire_obs.npz
wavesleuth-devito invert runs/circle_crossfire_obs.npz --method grid-search --candidate-grid-size 5 --refine-levels 1 --mismatch-mode differential --out runs/circle_crossfire_recon.json
wavesleuth-devito visualize-reconstruction runs/circle_crossfire_recon.json --out figures/circle_crossfire_recon.png
wavesleuth-devito score worlds/circle_crossfire.json runs/circle_crossfire_recon.json
```
## v0.3 improvements

v0.3 expands the playground from a better demo into a small experiment engine:

- `invert --radius-values`, `--anomaly-velocity-values`, `--search-radius`, and `--search-velocity` can search size and contrast, not just center.
- `simulate --noise-level`, `--receiver-dropout`, `--amplitude-jitter`, and `--time-jitter` create deterministic imperfect observations.
- `generate-world --acquisition-preset ring|top-only|left-right` adds more source/receiver layouts.
- `--boundary sponge --sponge-width N --sponge-strength X` enables a simple damping sponge. This is not a full PML.
- `visualize-uncertainty` turns candidate mismatch values into pseudo-probability diagnostics.
- `challenge` runs named budgeted reconstruction games such as `circle-easy`, `circle-noisy`, `circle-limited-angle`, and `circle-radius-velocity`.
- `leaderboard` scans challenge summaries and ranks them by budgeted score.
- `compare-acquisition` runs the same hidden object under multiple source/receiver layouts.
- `report` writes a small HTML experiment report.

Example v0.3 commands:

```bash
wavesleuth-devito demo --out-dir demo_output_v03 --quiet
wavesleuth-devito challenge circle-noisy --out-dir challenge_noisy --quiet
wavesleuth-devito leaderboard .
wavesleuth-devito generate-world --kind circle --acquisition-preset ring --boundary sponge --sponge-width 5 --sponge-strength 12 --out worlds/circle_ring.json
wavesleuth-devito compare-acquisition worlds/circle_ring.json --out-dir acq_compare --quiet
```

Radius/velocity search example:

```bash
wavesleuth-devito invert runs/circle_obs.npz \
  --method grid-search \
  --candidate-grid-size 5 \
  --refine-levels 1 \
  --mismatch-mode differential \
  --radius-values 0.09,0.12,0.15 \
  --anomaly-velocity-values 2.0,2.2,2.4 \
  --out runs/circle_recon_param_search.json
```
## v0.3.1 cleanup

v0.3.1 is a small polish release focused on diagnostics rather than new inversion behavior:

- uncertainty summaries now include `effective_candidates`, `center_effective_candidates`, and top-candidate probability mass diagnostics
- `visualize-uncertainty` handles single-center or degenerate candidate grids without Matplotlib identical-limit warnings
- leaderboard rows are rounded and sorted on rounded scores so tiny runtime jitter does not imply a meaningful ranking difference
- leaderboard rows include difficulty, experimental status, radius error, normalized center error, and a compact best-candidate summary
- challenge summaries now label `circle-radius-velocity` as hard/experimental and document why naive joint radius/velocity search can fail
- HTML reports include the uncertainty summary block
## v0.3.2 final cleanup

v0.3.2 is the final v0.3 cleanup patch before v0.4 work:

- challenge runs clean known challenge-owned outputs by default so stale files from a previous challenge do not linger in reused output directories
- challenge supports `--no-clean` / `--keep-existing` when you intentionally want to preserve old generated files
- default challenge scores no longer include wall-clock runtime, because first-run compilation and cache state make runtime too noisy for the main score
- runtime remains reported as a diagnostic field
- HTML reports backfill uncertainty diagnostics for older v0.3 reconstruction JSON files when candidate mismatches are available

## v0.4 staged radius/velocity search

v0.4 keeps the v0.3 joint grid search as a baseline, but adds a staged strategy for harder circle inversions where center, radius, and anomaly velocity are all uncertain.

The staged strategy is center-first:

1. search candidate centers using the reference radius and velocity
2. keep the top-K plausible centers
3. search radius and velocity only near those centers
4. optionally perform a final local center polish

This is designed to reduce the v0.3 failure mode where a small weak impostor anomaly can win the global joint objective before the center has been localized.

Example:

```bash
wavesleuth-devito invert runs/circle_obs.npz \
  --method staged-grid-search \
  --search-radius \
  --search-velocity \
  --candidate-grid-size 5 \
  --refine-levels 1 \
  --top-k-refine 5 \
  --out runs/circle_recon_staged.json
```

The old joint behavior is still available:

```bash
wavesleuth-devito invert runs/circle_obs.npz \
  --method grid-search \
  --search-strategy joint \
  --search-radius \
  --search-velocity \
  --out runs/circle_recon_joint.json
```

A new challenge compares the v0.4 staged method against the old hard radius/velocity baseline:

```bash
wavesleuth-devito challenge circle-radius-velocity-staged --out-dir challenge_rv_staged --quiet
wavesleuth-devito leaderboard challenge_rv challenge_rv_staged
```

## v0.4.1 reporting polish

v0.4.1 keeps the staged-search numerics unchanged and improves diagnostics around the hard radius/velocity challenge:

- circle scoring now reports anomaly-velocity error and relative velocity error when the true velocity is known
- challenge leaderboards display velocity-error fields
- reports backfill velocity diagnostics for older v0.4 reconstruction JSONs when possible
- staged uncertainty plots emphasize center-effective candidates because staged searches contain candidates from multiple phases

The staged radius/velocity method should be read as strong localization plus approximate contrast recovery, not perfect velocity inversion.

