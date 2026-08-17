# Changelog

## 1.0.0 - 2026-08-17

First stable release.

- Finalized the Devito-backed 2D inverse-wave playground.
- Included circle, ellipse, and coarse mask reconstruction baselines.
- Included staged parameter search, noise controls, acquisition presets,
  uncertainty summaries, blind challenges, active sensing, leaderboards,
  reports, artifact validation, and the canonical release suite.
- Promoted package and fresh artifact schema metadata to `1.0.0`.
- Added a reusable release gate with isolated wheel build/install validation.
- Finalized v1 documentation and the post-v1/v2 roadmap.
- Preserved the validated numerical behavior of v0.9.4.


## 0.9.4

- Fixed the active-demo release wrapper so package-version metadata resolves at runtime.
- Replaced the last hard-coded historical release-report sentence with a centralized, version-neutral note.
- Added static and runtime pre-v1 gates for version-symbol resolution and stale release wording.
- Kept all numerical simulation, inversion, challenge, and scoring behavior unchanged.


## v0.9.3

Final pre-v1.0 release candidate.

- Centralized fresh artifact schemas through `ARTIFACT_SCHEMA_VERSION`.
- Made challenge, blind, active, mask, and release-suite artifacts emit the current schema.
- Replaced version-specific generated descriptions with stable wording.
- Rewrote current-product documentation and added the post-v1 roadmap.
- Added audit-bundle Git-ignore rules and a release-hygiene helper.
- Hardened tests against stale exact-version assertions.
- Did not change numerical simulation, inversion, challenge calibration, or active policies.

## v0.9.2

Final pre-v1.0 release candidate. Fixes release-report version assertions,
active final-score reporting, schema/version consistency, and stale artifact
wording. No simulation or inversion numerics changed.

## 0.9.1

- Fixed active-demo artifact validation in release hardening.
- Made blind public worlds expose top-level `blind` and `answer_hidden` markers in addition to nested metadata.
- Kept numerical simulation and inversion behavior unchanged.


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
