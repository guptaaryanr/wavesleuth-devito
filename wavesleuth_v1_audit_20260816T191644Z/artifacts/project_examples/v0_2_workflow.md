# v0.2 workflow notes

The original MVP used raw trace L2 mismatch. That is intentionally naive, and it
can be fooled because the direct wave often dominates the trace energy. v0.2 adds
three features that make the toy inverse problem more honest and less ambiguous.

## 1. Differential inversion

The default inversion now subtracts a background simulation before comparing
traces:

```text
observed residual  = observed hidden-world traces - background traces
candidate residual = candidate traces - background traces
```

This emphasizes scattering from the anomaly instead of the direct arrival.

```bash
wavesleuth-devito invert runs/circle_obs.npz \
  --method grid-search \
  --mismatch-mode differential \
  --out runs/circle_recon.json
```

For comparison with the old behavior:

```bash
wavesleuth-devito invert runs/circle_obs.npz \
  --method grid-search \
  --mismatch-mode raw \
  --out runs/circle_recon_raw.json
```

## 2. Sequential crossfire acquisition

Generate a world with several sources fired one at a time:

```bash
wavesleuth-devito generate-world \
  --kind circle \
  --acquisition-preset crossfire \
  --out worlds/circle_crossfire.json

wavesleuth-devito simulate worlds/circle_crossfire.json \
  --shot-mode sequential \
  --out runs/circle_crossfire_obs.npz
```

The run file stores `receiver_traces` as `(shot, time, receiver)` when there is
more than one source.

## 3. Local refinement

After a coarse grid search, run one or more smaller local grids around the best
candidate:

```bash
wavesleuth-devito invert runs/circle_crossfire_obs.npz \
  --method grid-search \
  --candidate-grid-size 5 \
  --refine-levels 1 \
  --mismatch-mode differential \
  --out runs/circle_crossfire_recon.json
```

This is still brute force. It is just a more useful brute force baseline.
