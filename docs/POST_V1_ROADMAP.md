# Post-v1 roadmap

v1.0 is the completed compact playground. Post-v1 work should strengthen the
physics, inversion methods, and game layer without turning the project into a
large framework.

## v1.x

- better absorbing boundaries and PML experiments
- arrival-time, envelope, and phase-aware objectives
- improved scattered-arrival gating
- unknown-shape staged ellipse recovery
- ring, two-circle, crack, and layered-background inversion baselines
- regularized mask search and alternative optimizers
- stronger active experimental-design heuristics
- challenge bundle export/import, daily mystery, and sensor-economy modes

## Toward v2.0

- deterministic dataset generation
- small learned proposal models for center, size, and shape hints
- learned proposals followed by Devito verification and local refinement
- surrogate forward models for cheap candidate screening
- learned uncertainty and calibration studies
- a benchmark-lite suite comparing classical and learned baselines

The principle remains: learned guesser, physics verifier. Devito stays the
source of truth.
