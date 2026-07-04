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

Important outputs:

```text
active_demo/active_summary.json
active_demo/reports/active_report.html
active_demo/figures/active_progress.png
active_demo/figures/round_01_reconstruction.png
active_demo/figures/round_02_reconstruction.png
active_demo/figures/round_03_reconstruction.png
active_demo/runs/round_01_obs.npz
active_demo/runs/round_01_recon.json
```

The implementation re-simulates cumulative sequential shots each round. This is
simple and trustworthy for small worlds. Incremental shot caching is left for a
later optimization.
