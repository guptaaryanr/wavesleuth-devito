#!/usr/bin/env python3
"""Compatibility wrapper for tools/release_gate.py."""
from release_gate import main
if __name__ == "__main__":
    raise SystemExit(main())
