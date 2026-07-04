# v0.6 blind challenge workflow

v0.6 separates public observed data from secret answer metadata.

```bash
wavesleuth-devito challenge circle-easy --blind --out-dir blind_circle --quiet
```

Important files:

```text
blind_circle/challenge_manifest.json
blind_circle/worlds/circle-easy.json
blind_circle/public/circle-easy_public_world.json
blind_circle/runs/circle-easy_obs.npz
blind_circle/secret/circle-easy_secret_world.json
blind_circle/secret/answer_world.png
blind_circle/challenge_summary.json
```

The public `.npz` keeps receiver traces, time, sources, and receivers, but it
replaces the true velocity model with the background model and drops wavefield
snapshots. The embedded `world_json` is public metadata whose center is a neutral
placeholder, not the answer.

Score a challenge folder against its local secret answer:

```bash
wavesleuth-devito score-challenge blind_circle
```

Score a submitted reconstruction:

```bash
wavesleuth-devito score-challenge blind_circle \
  --reconstruction submissions/my_reconstruction.json
```

Known-shape hints may remain public in v0.6 because the current circle and
ellipse baselines need them. Fully unknown-shape blind challenges belong to a
later release.


## v0.6.1 integrity cleanup

v0.6.1 makes the answer-key hashes explicit:

```text
secret_world_sha256             file-byte SHA-256 of the saved secret world JSON
secret_world_file_sha256        same file-byte digest, named explicitly
secret_world_canonical_sha256   SHA-256 of compact sorted-key JSON
```

The private `secret/reconstruction_answer.png` figure intentionally overlays the
true anomaly and reconstruction. Public figures remain redacted.
