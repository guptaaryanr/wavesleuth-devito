# v0.5 workflow: first non-circle target

v0.5 adds richer hidden objects and the first non-circle inversion path: a conservative ellipse grid search.

```bash
wavesleuth-devito generate-world --kind ellipse --acquisition-preset crossfire --boundary sponge --sponge-width 5 --sponge-strength 12 --out worlds/ellipse.json
wavesleuth-devito simulate worlds/ellipse.json --shot-mode sequential --out runs/ellipse_obs.npz --quiet
wavesleuth-devito invert runs/ellipse_obs.npz --method ellipse-grid-search --candidate-grid-size 5 --refine-levels 1 --out runs/ellipse_recon.json --quiet
wavesleuth-devito visualize-world worlds/ellipse.json --out figures/ellipse_world.png
wavesleuth-devito visualize-reconstruction runs/ellipse_recon.json --out figures/ellipse_recon.png
wavesleuth-devito score worlds/ellipse.json runs/ellipse_recon.json
```

You can also generate harder non-circle worlds for simulation and visualization:

```bash
wavesleuth-devito generate-world --kind ring --out worlds/ring.json
wavesleuth-devito generate-world --kind two-circles --out worlds/two_circles.json
wavesleuth-devito generate-world --kind crack --out worlds/crack.json
wavesleuth-devito generate-world --kind circle-layered --out worlds/circle_layered.json
```

Only circle and ellipse have parametric reconstruction baselines in v0.5. Other v0.5 targets are there to broaden the world gallery and prepare future inversion work.
