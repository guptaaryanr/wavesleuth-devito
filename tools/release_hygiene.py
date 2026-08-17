#!/usr/bin/env python3
"""Report or clean generated WaveSleuth release-audit artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PATTERNS = (
    "wavesleuth_v1_audit_*",
    "release_suite",
    "reports/release_report*.html",
)


def tracked_audits(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--", "wavesleuth_v1_audit_*"],
        check=False,
        stdout=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return []
    return [item.decode() for item in proc.stdout.split(b"\0") if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--untrack-audits", action="store_true")
    parser.add_argument("--delete-generated", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()

    tracked = tracked_audits(repo)
    print(f"tracked audit paths: {len(tracked)}")
    for path in tracked:
        print(f"  {path}")

    if args.untrack_audits and tracked:
        subprocess.run(
            ["git", "-C", str(repo), "rm", "-r", "--cached", "--ignore-unmatch", "--", *tracked],
            check=True,
        )
        print("Removed audit artifacts from the Git index; files were kept on disk.")

    if args.delete_generated:
        for pattern in PATTERNS:
            for path in repo.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                print(f"deleted {path.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
