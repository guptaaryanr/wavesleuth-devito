# Quickstart

From the repository root after installation:

```bash
wavesleuth-devito generate-world --kind circle --out worlds/circle.json
wavesleuth-devito visualize-world worlds/circle.json --out figures/circle_world.png
wavesleuth-devito simulate worlds/circle.json --out runs/circle_obs.npz
wavesleuth-devito visualize-run runs/circle_obs.npz --out figures/circle_traces.png
wavesleuth-devito invert runs/circle_obs.npz --method grid-search --candidate-grid-size 5 --out runs/circle_recon.json
wavesleuth-devito visualize-reconstruction runs/circle_recon.json --out figures/circle_recon.png
wavesleuth-devito score worlds/circle.json runs/circle_recon.json
```

For a smaller one-command pipeline:

```bash
wavesleuth-devito demo --out-dir demo_output
```

The generated files are intentionally plain JSON, NPZ, and PNG so you can inspect and modify them without notebooks.
