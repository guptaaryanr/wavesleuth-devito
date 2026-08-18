#!/usr/bin/env bash
# WaveSleuth-Devito v1.0 readiness audit collector.
#
# Run from the repository root:
#   bash collect_wavesleuth_v1_audit.sh
#
# Optional:
#   WAVESLEUTH_AUDIT_ROOT=my_audit bash collect_wavesleuth_v1_audit.sh
#   WAVESLEUTH_AUDIT_FAST=1 bash collect_wavesleuth_v1_audit.sh
#
# Output:
#   wavesleuth_v1_audit_<timestamp>/
#   wavesleuth_v1_audit_<timestamp>.zip

set -u
set -o pipefail

AUDIT_ROOT="${WAVESLEUTH_AUDIT_ROOT:-wavesleuth_v1_audit_$(date -u +%Y%m%dT%H%M%SZ)}"
FAST="${WAVESLEUTH_AUDIT_FAST:-0}"

mkdir -p "$AUDIT_ROOT"/{logs,meta,reports,world_gallery,release,blind,active,artifacts}

COMMAND_STATUS="$AUDIT_ROOT/meta/command_status.tsv"
: > "$COMMAND_STATUS"

echo "WaveSleuth-Devito v1.0 readiness audit"
echo "Output folder: $AUDIT_ROOT"
echo

run_cmd() {
  local name="$1"
  shift
  local logfile="$AUDIT_ROOT/logs/${name}.log"
  echo "[$(date -u +%H:%M:%S)] RUN $name: $*" | tee -a "$AUDIT_ROOT/logs/audit_progress.log"
  {
    echo "\$ $*"
    echo
    "$@"
  } >"$logfile" 2>&1
  local rc=$?
  printf "%s\t%s\t%s\n" "$name" "$rc" "$*" >> "$COMMAND_STATUS"
  if [ "$rc" -ne 0 ]; then
    echo "  -> FAILED rc=$rc, see $logfile" | tee -a "$AUDIT_ROOT/logs/audit_progress.log"
  else
    echo "  -> ok" | tee -a "$AUDIT_ROOT/logs/audit_progress.log"
  fi
  return 0
}

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp -R "$src" "$dst"
  fi
}

# ---------------------------------------------------------------------------
# 0. Environment and source snapshot
# ---------------------------------------------------------------------------

{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pwd=$(pwd)"
  echo "uname=$(uname -a || true)"
  echo "python=$(python --version 2>&1 || true)"
  echo "which_python=$(command -v python || true)"
  echo "wavesleuth_version=$(wavesleuth-devito --version 2>&1 || true)"
} > "$AUDIT_ROOT/meta/environment.txt"

run_cmd "python_version" python --version
run_cmd "pip_freeze" python -m pip freeze
run_cmd "wavesleuth_version" wavesleuth-devito --version
run_cmd "cli_help" wavesleuth-devito --help

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  run_cmd "git_status" git status --short
  run_cmd "git_log" git log --oneline -n 25
  run_cmd "git_diff_stat" git diff --stat
  run_cmd "git_diff" git diff
fi

# Copy lightweight project metadata/docs if present.
for p in pyproject.toml README.md CHANGELOG.md LICENSE docs examples .github/workflows; do
  if [ -e "$p" ]; then
    copy_if_exists "$p" "$AUDIT_ROOT/artifacts/project_$p"
  fi
done

# ---------------------------------------------------------------------------
# 1. Tests and doctor checks
# ---------------------------------------------------------------------------

run_cmd "pytest" python -m pytest
run_cmd "doctor" wavesleuth-devito doctor
run_cmd "doctor_try_devito" wavesleuth-devito doctor --try-devito
run_cmd "self_test" wavesleuth-devito self-test
run_cmd "self_test_try_devito" wavesleuth-devito self-test --try-devito

# CLI surface snapshot.
for cmd in \
  generate-world simulate invert visualize-world visualize-run visualize-reconstruction \
  visualize-uncertainty score challenge leaderboard active-demo active-leaderboard \
  score-challenge validate challenge-suite release-report doctor report
do
  run_cmd "help_${cmd}" wavesleuth-devito "$cmd" --help
done

# ---------------------------------------------------------------------------
# 2. World gallery, no heavy inversion
# ---------------------------------------------------------------------------

WORLD_KINDS=(
  circle rectangle layered blobs
  ellipse ring two-circles crack circle-layered mask-blocks
)

for kind in "${WORLD_KINDS[@]}"; do
  safe_kind="${kind//-/_}"
  run_cmd "gallery_generate_${safe_kind}" \
    wavesleuth-devito generate-world --kind "$kind" --acquisition-preset crossfire --out "$AUDIT_ROOT/world_gallery/${kind}.json"

  run_cmd "gallery_visualize_${safe_kind}" \
    wavesleuth-devito visualize-world "$AUDIT_ROOT/world_gallery/${kind}.json" --out "$AUDIT_ROOT/world_gallery/${kind}.png"
done

# ---------------------------------------------------------------------------
# 3. Standard release challenge suite
# ---------------------------------------------------------------------------

run_cmd "challenge_suite" \
  wavesleuth-devito challenge-suite --out-dir "$AUDIT_ROOT/release/release_suite" --quiet

run_cmd "validate_release_suite" \
  wavesleuth-devito validate \
    "$AUDIT_ROOT/release/release_suite/circle-easy" \
    "$AUDIT_ROOT/release/release_suite/ellipse-easy" \
    "$AUDIT_ROOT/release/release_suite/circle-radius-velocity-staged" \
    "$AUDIT_ROOT/release/release_suite/mask-cell-easy"

run_cmd "leaderboard_release_suite" \
  wavesleuth-devito leaderboard \
    "$AUDIT_ROOT/release/release_suite/circle-easy" \
    "$AUDIT_ROOT/release/release_suite/ellipse-easy" \
    "$AUDIT_ROOT/release/release_suite/circle-radius-velocity-staged" \
    "$AUDIT_ROOT/release/release_suite/mask-cell-easy"

# Copy standard suite summary/report to top-level reports.
copy_if_exists "$AUDIT_ROOT/release/release_suite/release_suite_summary.json" "$AUDIT_ROOT/reports/release_suite_summary.json"
copy_if_exists "$AUDIT_ROOT/release/release_suite/release_suite_report.html" "$AUDIT_ROOT/reports/release_suite_report.html"

# ---------------------------------------------------------------------------
# 4. Blind challenge integrity check
# ---------------------------------------------------------------------------

run_cmd "blind_challenge_ellipse" \
  wavesleuth-devito challenge ellipse-easy --blind --out-dir "$AUDIT_ROOT/blind/challenge_ellipse_blind" --quiet

run_cmd "score_challenge_blind" \
  wavesleuth-devito score-challenge "$AUDIT_ROOT/blind/challenge_ellipse_blind"

run_cmd "validate_blind_challenge" \
  wavesleuth-devito validate "$AUDIT_ROOT/blind/challenge_ellipse_blind"

# Hash check for secret world.
python - "$AUDIT_ROOT" > "$AUDIT_ROOT/blind/secret_hash_check.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
challenge = root / "blind" / "challenge_ellipse_blind"
manifest_path = challenge / "challenge_manifest.json"
result = {
    "challenge_dir": str(challenge),
    "manifest_exists": manifest_path.exists(),
    "ok": False,
}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    result["manifest"] = {
        k: manifest.get(k)
        for k in [
            "challenge",
            "blind",
            "schema_version",
            "secret_world_sha256",
            "secret_world_file_sha256",
            "secret_world_canonical_sha256",
            "secret_world_path",
        ]
        if k in manifest
    }
    secret_rel = manifest.get("secret_world_path")
    candidates = []
    if secret_rel:
        candidates.append(challenge / secret_rel)
        candidates.append(root / secret_rel)
        candidates.append(Path(secret_rel))
    candidates.extend((challenge / "secret").glob("*_secret_world.json"))
    secret_path = next((p for p in candidates if p.exists()), None)
    result["secret_world_path_detected"] = str(secret_path) if secret_path else None
    if secret_path:
        file_hash = hashlib.sha256(secret_path.read_bytes()).hexdigest()
        result["actual_file_sha256"] = file_hash
        result["matches_manifest_file_hash"] = file_hash == manifest.get("secret_world_file_sha256")
        result["matches_manifest_main_hash"] = file_hash == manifest.get("secret_world_sha256")
        result["ok"] = bool(result["matches_manifest_file_hash"] and result["matches_manifest_main_hash"])
print(json.dumps(result, indent=2, sort_keys=True))
PY

# Public redaction check for blind observed run.
python - "$AUDIT_ROOT" > "$AUDIT_ROOT/blind/public_redaction_check.json" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
challenge = root / "blind" / "challenge_ellipse_blind"
runs = sorted((challenge / "runs").glob("*_obs.npz"))
result = {"challenge_dir": str(challenge), "run_files": [str(p) for p in runs], "ok": False}
if runs:
    p = runs[0]
    with np.load(p, allow_pickle=False) as data:
        result["arrays"] = sorted(data.files)
        if "velocity_model" in data:
            vm = data["velocity_model"]
            result["velocity_model_min"] = float(vm.min())
            result["velocity_model_max"] = float(vm.max())
            result["velocity_model_is_constant"] = bool(float(vm.min()) == float(vm.max()))
        if "final_wavefield" in data:
            result["final_wavefield_shape"] = list(data["final_wavefield"].shape)
        if "snapshots" in data:
            result["snapshots_shape"] = list(data["snapshots"].shape)
        if "world_json" in data:
            world = json.loads(str(data["world_json"].item()))
            result["world_blind"] = world.get("blind")
            result["answer_hidden"] = world.get("answer_hidden")
            result["public_anomaly"] = world.get("medium", {}).get("anomaly", {})
    result["ok"] = bool(
        result.get("velocity_model_is_constant") is True
        and result.get("world_blind") is True
        and result.get("answer_hidden") is True
    )
print(json.dumps(result, indent=2, sort_keys=True))
PY

# ---------------------------------------------------------------------------
# 5. Active sensing demos and active leaderboard
# ---------------------------------------------------------------------------

if [ "$FAST" = "1" ]; then
  ACTIVE_ARGS=(--rounds 2 --candidate-grid-size 5 --refine-levels 0 --quiet)
else
  ACTIVE_ARGS=(--rounds 3 --candidate-grid-size 5 --refine-levels 1 --quiet)
fi

run_cmd "active_uncertainty" \
  wavesleuth-devito active-demo --out-dir "$AUDIT_ROOT/active/active_uncertainty" --strategy uncertainty "${ACTIVE_ARGS[@]}"

run_cmd "active_spread" \
  wavesleuth-devito active-demo --out-dir "$AUDIT_ROOT/active/active_spread" --strategy spread "${ACTIVE_ARGS[@]}"

run_cmd "active_opposite" \
  wavesleuth-devito active-demo --out-dir "$AUDIT_ROOT/active/active_opposite" --strategy opposite-best "${ACTIVE_ARGS[@]}"

run_cmd "active_ellipse" \
  wavesleuth-devito active-demo --kind ellipse --out-dir "$AUDIT_ROOT/active/active_ellipse" --strategy uncertainty "${ACTIVE_ARGS[@]}"

run_cmd "active_leaderboard" \
  wavesleuth-devito active-leaderboard \
    "$AUDIT_ROOT/active/active_uncertainty" \
    "$AUDIT_ROOT/active/active_spread" \
    "$AUDIT_ROOT/active/active_opposite" \
    "$AUDIT_ROOT/active/active_ellipse"

run_cmd "validate_active" \
  wavesleuth-devito validate \
    "$AUDIT_ROOT/active/active_uncertainty" \
    "$AUDIT_ROOT/active/active_spread" \
    "$AUDIT_ROOT/active/active_opposite" \
    "$AUDIT_ROOT/active/active_ellipse"

# Extra trace-shape audit for active outputs.
python - "$AUDIT_ROOT" > "$AUDIT_ROOT/active/active_trace_shape_audit.json" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
active_root = root / "active"
rows = []
ok = True
for run_dir in sorted(active_root.glob("active_*")):
    for npz in sorted((run_dir / "runs").glob("round_*_obs.npz")):
        with np.load(npz, allow_pickle=False) as data:
            shape = tuple(data["receiver_traces"].shape)
        row = {"active_run": run_dir.name, "file": str(npz.relative_to(root)), "shape": list(shape)}
        rows.append(row)
        if len(shape) != 3:
            ok = False
            row["error"] = "active sequential traces should be 3D: (shot, time, receiver)"
print(json.dumps({"ok": ok, "rows": rows}, indent=2, sort_keys=True))
PY

# ---------------------------------------------------------------------------
# 6. Combined release reports
# ---------------------------------------------------------------------------

run_cmd "release_report_challenges_only" \
  wavesleuth-devito release-report \
    --out "$AUDIT_ROOT/reports/release_report_challenges_only.html" \
    --challenge-paths \
      "$AUDIT_ROOT/release/release_suite/circle-easy" \
      "$AUDIT_ROOT/release/release_suite/ellipse-easy" \
      "$AUDIT_ROOT/release/release_suite/circle-radius-velocity-staged" \
      "$AUDIT_ROOT/release/release_suite/mask-cell-easy"

run_cmd "release_report_full" \
  wavesleuth-devito release-report \
    --out "$AUDIT_ROOT/reports/release_report_full.html" \
    --challenge-paths \
      "$AUDIT_ROOT/release/release_suite/circle-easy" \
      "$AUDIT_ROOT/release/release_suite/ellipse-easy" \
      "$AUDIT_ROOT/release/release_suite/circle-radius-velocity-staged" \
      "$AUDIT_ROOT/release/release_suite/mask-cell-easy" \
      "$AUDIT_ROOT/blind/challenge_ellipse_blind" \
    --active-paths \
      "$AUDIT_ROOT/active/active_uncertainty" \
      "$AUDIT_ROOT/active/active_spread" \
      "$AUDIT_ROOT/active/active_opposite" \
      "$AUDIT_ROOT/active/active_ellipse"

# ---------------------------------------------------------------------------
# 7. Machine-readable audit summary
# ---------------------------------------------------------------------------

python - "$AUDIT_ROOT" > "$AUDIT_ROOT/audit_summary.json" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
status_path = root / "meta" / "command_status.tsv"

commands = []
if status_path.exists():
    for line in status_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) == 3:
            commands.append({"name": parts[0], "returncode": int(parts[1]), "command": parts[2]})

def load_json(path):
    p = root / path
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception as exc:
            return {"error": f"could not parse {path}: {exc}"}
    return None

summary = {
    "audit_root": str(root),
    "commands_total": len(commands),
    "commands_failed": [c for c in commands if c["returncode"] != 0],
    "release_suite_summary": load_json("reports/release_suite_summary.json"),
    "blind_secret_hash_check": load_json("blind/secret_hash_check.json"),
    "blind_public_redaction_check": load_json("blind/public_redaction_check.json"),
    "active_trace_shape_audit": load_json("active/active_trace_shape_audit.json"),
}

# Extract compact active summaries.
active_summaries = {}
for p in sorted((root / "active").glob("active_*/active_summary.json")):
    try:
        data = json.loads(p.read_text())
        active_summaries[p.parent.name] = {
            "kind": data.get("kind"),
            "strategy": data.get("strategy"),
            "rounds": data.get("rounds_requested") or data.get("rounds"),
            "initial_reconstruction_score": data.get("initial_reconstruction_score"),
            "final_reconstruction_score": data.get("final_reconstruction_score"),
            "score_delta": data.get("score_delta"),
            "source_history": data.get("source_history"),
        }
    except Exception as exc:
        active_summaries[p.parent.name] = {"error": str(exc)}
summary["active_summaries"] = active_summaries

# Extract release-suite validation.
release = summary.get("release_suite_summary") or {}
summary["release_validation_ok"] = (release.get("validation") or {}).get("ok")
summary["release_leaderboard"] = release.get("leaderboard")

print(json.dumps(summary, indent=2, sort_keys=True))
PY

# ---------------------------------------------------------------------------
# 8. Checksums and zip bundle
# ---------------------------------------------------------------------------

(
  cd "$AUDIT_ROOT"
  find . -type f -not -path "./checksums.sha256" -print0 | sort -z | xargs -0 sha256sum > checksums.sha256
)

if command -v zip >/dev/null 2>&1; then
  zip -qr "${AUDIT_ROOT}.zip" "$AUDIT_ROOT"
  sha256sum "${AUDIT_ROOT}.zip" > "${AUDIT_ROOT}.zip.sha256"
else
  tar -czf "${AUDIT_ROOT}.tar.gz" "$AUDIT_ROOT"
  sha256sum "${AUDIT_ROOT}.tar.gz" > "${AUDIT_ROOT}.tar.gz.sha256"
fi

echo
echo "Audit complete."
echo "Folder: $AUDIT_ROOT"
if [ -f "${AUDIT_ROOT}.zip" ]; then
  echo "Archive: ${AUDIT_ROOT}.zip"
  echo "Archive checksum: ${AUDIT_ROOT}.zip.sha256"
else
  echo "Archive: ${AUDIT_ROOT}.tar.gz"
  echo "Archive checksum: ${AUDIT_ROOT}.tar.gz.sha256"
fi
echo
echo "Key files:"
echo "  $AUDIT_ROOT/audit_summary.json"
echo "  $AUDIT_ROOT/reports/release_report_full.html"
echo "  $AUDIT_ROOT/reports/release_suite_summary.json"
echo "  $AUDIT_ROOT/logs/pytest.log"
echo "  $AUDIT_ROOT/meta/command_status.tsv"
echo
echo "Failed commands, if any:"
awk -F '\t' '$2 != 0 {print "  " $1 " rc=" $2 " :: " $3}' "$COMMAND_STATUS" || true
