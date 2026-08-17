# Final pre-v1 release candidate

0.9.4 changes only release-layer behavior:

- one current schema constant for every fresh artifact
- version-neutral suite and challenge wording
- current-product documentation
- audit-bundle Git hygiene
- regression tests against stale exact-version assertions

It does not alter the Devito solver, inversion objectives, challenge
calibration, active policies, or expected leaderboard results.

Run the final gate from a clean committed tree:

```bash
python -m pytest
wavesleuth-devito doctor --try-devito
wavesleuth-devito self-test --try-devito
bash collect_wavesleuth_v1_audit.sh
```
