# v0.7 active sensing workflow

v0.7 adds a small active-sensing loop. It is not an optimal experimental-design
solver; it is a deterministic, inspectable strategy that uses the current
reconstruction and uncertainty to choose the next source location.

Run the default active circle demo:

```bash
wavesleuth-devito active-demo --out-dir active_demo --quiet
```

Useful options:

```bash
wavesleuth-devito active-demo \
  --kind circle \
  --rounds 3 \
  --candidate-grid-size 5 \
  --refine-levels 1 \
  --strategy uncertainty \
  --out-dir active_circle
```

Try the ellipse version:

```bash
wavesleuth-devito active-demo --kind ellipse --out-dir active_ellipse --quiet
```

Compare strategies after running multiple active demos:

```bash
wavesleuth-devito active-demo --out-dir active_uncertainty --strategy uncertainty --quiet
wavesleuth-devito active-demo --out-dir active_spread --strategy spread --quiet
wavesleuth-devito active-demo --out-dir active_opposite --strategy opposite-best --quiet
wavesleuth-devito active-leaderboard active_uncertainty active_spread active_opposite
```

Important outputs:

```text
active_demo/active_summary.json
active_demo/reports/active_report.html
active_demo/figures/active_progress.png
active_demo/figures/active_source_layout.png
active_demo/figures/round_01_reconstruction.png
active_demo/figures/round_02_reconstruction.png
active_demo/figures/round_03_reconstruction.png
active_demo/runs/round_01_obs.npz
active_demo/runs/round_01_recon.json
```

v0.7.2 keeps the public `simulate_world()` single-shot shape backward-compatible,
but active-demo output files are standardized to `(shot, time, receiver)`,
including one-shot rounds.

The implementation re-simulates cumulative sequential shots each round. This is
simple and trustworthy for small worlds. Incremental shot caching is left for a
later optimization.
## v0.7.2 trace-shape note

The public `simulate_world()` API remains backward-compatible: a single-shot run
returns/saves receiver traces as `(time, receiver)`. Active-demo artifacts are
standardized after simulation so active rounds always save traces as
`(shot, time, receiver)`, including round 1 with one source.
