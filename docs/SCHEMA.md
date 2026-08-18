# Artifact schema

The current artifact schema version is `1.0.0`. Fresh artifacts use the same
value as the package version through `ARTIFACT_SCHEMA_VERSION` in
`wavesleuth_devito.metadata`.

## Compatibility policy

- Fresh worlds, reconstructions, challenge manifests, challenge summaries,
  active summaries, and release suite summaries emit the current schema.
- Supported older artifacts remain readable.
- Normal validation identifies historical schemas without treating them as
  errors.
- `validate --strict` warns when regeneration would move an artifact to the
  current schema.
- Artifacts produced by a newer schema are reported explicitly.

## Core artifacts

### World JSON

Contains grid, physical extent, medium parameters, anomaly description,
acquisition geometry, and simulation settings.

### Run NPZ

Contains receiver traces, time, velocity model or redacted background model,
source and receiver coordinates, optional wavefield products, and serialized
world metadata.

Sequential active artifacts use a stable trace layout:

```text
(shot, time, receiver)
```

Ordinary single shot `simulate_world()` remains backward compatible and may
return `(time, receiver)` in memory.

### Reconstruction JSON

Contains method, target kind, candidates or selected cells, objective settings,
best candidate, mismatch information, uncertainty summaries where available,
and physical scoring when answer metadata is available.

### Challenge manifest

Contains public paths, blind state metadata, difficulty, schema, and secret
hashes when blind mode is enabled.

### Challenge summary

Uses:

```text
physical_score
challenge_score
```

`score` remains a compatibility alias for the physical score.

### Active summary

Contains strategy, source history, per round trace metadata, per round physical
scores and uncertainty diagnostics, final physical score, and score change.

### Release suite summary

Uses the stable suite identifier:

```text
standard-challenge-suite
```

Package/schema versions are stored separately.

## Blind artifacts

Public blind worlds expose top level markers:

```json
{
  "blind": true,
  "answer_hidden": true,
  "blind_schema_version": "1.0.0"
}
```

The nested `blind_public_metadata` object retains richer redaction notes.
Secret file byte and canonical hashes are stored separately.
