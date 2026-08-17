# WaveSleuth-Devito v1.0.0 release notes

v1.0.0 is the first stable release of the compact Devito-backed inverse-wave
playground.

## Included

- constant-density 2D acoustic simulation through Devito
- simultaneous and sequential shots, acquisition presets, and data perturbations
- circle, rectangle, layered, blobs, ellipse, ring, two-circle, crack,
  circle-layered, and mask-blocks world generation
- circle joint/staged search, known-shape ellipse center search, and greedy
  coarse mask-cell reconstruction
- mismatch-derived uncertainty summaries
- open and blind challenges, physical and budgeted scores, and leaderboards
- multi-round active source-selection demonstrations
- HTML reports, artifact validation, environment doctor, canonical release
  suite, and isolated wheel-install release smoke

## Canonical regression references

| Challenge | Reference IoU |
|---|---:|
| `mask-cell-easy` | 1.000 |
| `circle-radius-velocity-staged` | 0.895 |
| `circle-easy` | 0.800 |
| `ellipse-easy` | 0.787 |

These are tiny deterministic regression references, not general accuracy
claims. v1.0.0 intentionally preserves the numerical behavior of the clean
v0.9.4 gate.
