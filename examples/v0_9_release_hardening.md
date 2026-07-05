# v0.9 release hardening workflow

Run lightweight environment checks:

```bash
wavesleuth-devito doctor
wavesleuth-devito doctor --try-devito
```

Validate existing artifacts:

```bash
wavesleuth-devito validate challenge_easy challenge_ellipse challenge_mask active_uncertainty
```

Run the standard challenge suite:

```bash
wavesleuth-devito challenge-suite --out-dir release_suite --quiet
```

Generate a release report from existing outputs:

```bash
wavesleuth-devito release-report \
  --out reports/release_report.html \
  --challenge-paths challenge_easy challenge_ellipse challenge_rv_staged challenge_mask \
  --active-paths active_uncertainty active_spread active_opposite
```

v0.9 does not change the numerical methods. It is about making the project
cleaner to validate, explain, and freeze before v1.0 polish.
