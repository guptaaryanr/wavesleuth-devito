# v1.0 release verification

```bash
python tools/release_gate.py --with-devito
bash collect_wavesleuth_v1_audit.sh
```

Expected audit state:

```text
commands_failed: []
release_validation_ok: true
blind_secret_hash_check.ok: true
blind_public_redaction_check.ok: true
active_trace_shape_audit.ok: true
```
