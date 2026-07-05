# v0.8 coarse mask inversion workflow

v0.8 adds the first coarse mask/image reconstruction baseline. The new
`mask-blocks` world represents the hidden target as active cells on a small
rectangular grid. The `cell-search` inversion greedily adds one cell at a time
and verifies each candidate with the Devito forward model.

```bash
wavesleuth-devito generate-world --kind mask-blocks --acquisition-preset crossfire --out worlds/mask_blocks.json
wavesleuth-devito visualize-world worlds/mask_blocks.json --out figures/mask_blocks_world.png
wavesleuth-devito simulate worlds/mask_blocks.json --shot-mode sequential --out runs/mask_blocks_obs.npz --quiet
wavesleuth-devito invert runs/mask_blocks_obs.npz --method cell-search --cell-grid-size 6 --max-active-cells 5 --out runs/mask_blocks_recon.json --quiet
wavesleuth-devito visualize-reconstruction runs/mask_blocks_recon.json --out figures/mask_blocks_recon.png
wavesleuth-devito score worlds/mask_blocks.json runs/mask_blocks_recon.json
```

Or run the challenge:

```bash
wavesleuth-devito challenge mask-cell-easy --out-dir challenge_mask --quiet
wavesleuth-devito leaderboard challenge_mask
```
