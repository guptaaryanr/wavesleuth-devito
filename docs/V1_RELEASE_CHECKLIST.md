# v1.0 release verification checklist

- [ ] `python -m pip install -e ".[devito,test]"` succeeds
- [ ] `python tools/release_gate.py --with-devito` passes
- [ ] the isolated wheel build/install smoke passes
- [ ] the standard challenge suite validates
- [ ] fresh open, blind, and active artifacts use schema `1.0.0`
- [ ] blind hashes and public redaction checks pass
- [ ] active traces remain `(shot, time, receiver)`
- [ ] the combined release report has overall validation `True`
- [ ] `bash collect_wavesleuth_v1_audit.sh` reports no failed commands
- [ ] the working tree is clean and tag `v1.0.0` is created
