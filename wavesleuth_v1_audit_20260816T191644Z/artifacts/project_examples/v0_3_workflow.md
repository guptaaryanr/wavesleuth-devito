# v0.3 workflow

v0.3 adds richer search, noisy observations, uncertainty maps, challenge mode,
acquisition comparison, a simple sponge damping option, and HTML reports.

## Search radius and velocity

```bash
wavesleuth-devito invert runs/circle_obs.npz \
  --method grid-search \
  --candidate-grid-size 5 \
  --refine-levels 1 \
  --mismatch-mode differential \
  --radius-values 0.09,0.12,0.15 \
  --anomaly-velocity-values 2.0,2.2,2.4 \
  --out runs/circle_recon_rv.json
```

## Noisy observations

```bash
wavesleuth-devito simulate worlds/circle.json \
  --shot-mode sequential \
  --noise-level 0.04 \
  --receiver-dropout 0.05 \
  --amplitude-jitter 0.03 \
  --noise-seed 123 \
  --out runs/circle_noisy_obs.npz
```

## Uncertainty map and report

```bash
wavesleuth-devito visualize-uncertainty runs/circle_recon.json \
  --out figures/circle_uncertainty.png

wavesleuth-devito report runs/circle_recon.json \
  --out reports/circle_report.html
```

## Challenge mode

```bash
wavesleuth-devito challenge circle-easy --out-dir challenge_easy --quiet
wavesleuth-devito challenge circle-noisy --out-dir challenge_noisy --quiet
wavesleuth-devito challenge circle-radius-velocity --out-dir challenge_rv --quiet
```

## Acquisition comparison

```bash
wavesleuth-devito compare-acquisition \
  --presets single,crossfire,ring \
  --out-dir acquisition_compare \
  --quiet
```

## Simple sponge damping

The sponge is a basic edge damping layer, not a tuned PML.

```bash
wavesleuth-devito simulate worlds/circle.json \
  --sponge-width 8 \
  --sponge-strength 1.0 \
  --out runs/circle_sponge_obs.npz
```
