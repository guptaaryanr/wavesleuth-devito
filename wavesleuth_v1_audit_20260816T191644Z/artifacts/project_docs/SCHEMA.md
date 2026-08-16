# WaveSleuth-Devito artifact schema notes

v0.9 is the first schema-audit release. The project still uses plain JSON and
NPZ files, but the following naming conventions are now preferred.

## World JSON

A world JSON contains:

- `name`
- `grid`
- `medium`
- `acquisition`
- `simulation`

The hidden object lives under `medium.anomaly`. Supported generated target
families include circles, ellipses, rings, two-circles, cracks, layered variants,
blobs, and mask-blocks.

## Run NPZ

A run NPZ contains receiver traces, time, acquisition coordinates, velocity
metadata, and serialized public world metadata.

Trace layouts:

- ordinary one-shot/simultaneous traces may be `(time, receiver)`
- sequential multi-shot traces use `(shot, time, receiver)`
- active-demo artifacts standardize cumulative rounds to `(shot, time, receiver)`

Blind public runs replace the true velocity model with the background model and
remove final wavefield/snapshot arrays.

## Reconstruction JSON

Preferred fields:

- `method`
- `best_candidate`
- `objective`
- `candidate_grid` or search metadata
- `physical_score`
- `challenge_score` when the reconstruction belongs to a challenge

Backward compatibility:

- older files may use `score` for the physical reconstruction score
- v0.9 readers normalize this to `physical_score`

## Challenge directory

A challenge directory contains:

```text
challenge_summary.json
challenge_manifest.json
worlds/
runs/
figures/
reports/
secret/        # blind/local answer-key mode only
public/        # blind public metadata only
```

Preferred score names:

- `physical_score`: reconstruction quality, such as IoU and center error
- `challenge_score`: budgeted game score, including forward-run and sensing cost

## v0.9 validation commands

```bash
wavesleuth-devito doctor
wavesleuth-devito validate challenge_easy challenge_mask runs/mask_blocks_obs.npz
wavesleuth-devito release-report --out reports/release_report.html --challenge-paths challenge_easy challenge_mask
```
