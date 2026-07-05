# Changelog

## v0.9.0

Release-hardening pass.

- Added `doctor` command for environment/package sanity checks.
- Added `validate` command for worlds, runs, reconstructions, challenge outputs, and active-demo outputs.
- Added `challenge-suite` command for running the standard release challenge set.
- Added `release-report` command for compact HTML release reports.
- Added `wavesleuth_devito.release` helpers for schema normalization and artifact validation.
- Added artifact schema documentation under `docs/SCHEMA.md`.
- Clarified `physical_score` vs `challenge_score` naming.
- Updated challenge manifest/summary schema version for newly generated outputs.
- No numerical solver, inversion, active-sensing, or mask-search algorithm changes.

## v0.8.3

Finalized the first coarse mask-cell challenge calibration.

## v0.7.2

Finalized active-sensing artifact shape compatibility.

## v0.6.1

Finalized blind challenge integrity and answer-key visualization cleanup.
