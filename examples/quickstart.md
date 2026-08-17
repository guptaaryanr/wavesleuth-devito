# Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[devito,test]"

wavesleuth-devito demo --out-dir demo_output --quiet
wavesleuth-devito challenge-suite --out-dir release_suite --quiet
wavesleuth-devito doctor --try-devito
```

Inspect the standard suite:

```bash
wavesleuth-devito leaderboard \
  release_suite/circle-easy \
  release_suite/ellipse-easy \
  release_suite/circle-radius-velocity-staged \
  release_suite/mask-cell-easy
```

Run a blind challenge:

```bash
wavesleuth-devito challenge ellipse-easy \
  --blind \
  --out-dir challenge_ellipse_blind \
  --quiet
wavesleuth-devito score-challenge challenge_ellipse_blind
```

Run active sensing:

```bash
wavesleuth-devito active-demo \
  --strategy uncertainty \
  --rounds 3 \
  --out-dir active_uncertainty \
  --quiet
```

Run the release gate:

```bash
python tools/release_gate.py --with-devito
```
