# Roadmap from v0.9 to v1.0

v0.9 freezes the broad feature set and focuses on hardening. The next major
milestone is v1.0, a coherent local playground release.

## Current pillars

- Devito-backed 2D acoustic forward simulation
- circle and ellipse parametric inversion
- staged circle radius/velocity search
- greedy coarse mask-cell inversion
- active sensing demos
- open and blind challenge modes
- leaderboard, reports, validation helpers

## v1.0 release criteria

- standard challenge suite runs cleanly on a local CPU
- public README reflects actual commands and limitations
- artifact schemas are documented
- core demos and reports are reproducible
- Devito-absent pytest path skips simulation tests cleanly
- Devito-present path passes the tiny simulation smoke tests
- no new major algorithms added during final release polishing

## Post-v1.0 directions

- richer active sensing policies
- stronger non-parametric mask inversion
- learned proposal models with Devito verification
- better absorbing boundaries
- daily/boss challenge modes
