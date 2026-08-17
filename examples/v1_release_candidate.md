# v1 release-candidate verification

From the repository root after applying v0.9.2:

```bash
python -m pytest
wavesleuth-devito --version
wavesleuth-devito doctor --try-devito
wavesleuth-devito self-test --try-devito
bash collect_wavesleuth_v1_audit.sh
```

The final audit should contain no failed commands and should validate challenge,
blind, and active-sensing outputs before the version is changed to 1.0.0.
