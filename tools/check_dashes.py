#!/usr/bin/env python3
"""Reject em dash (U+2014) and en dash (U+2013) in repository text files.

The team writes prose with ASCII punctuation only. Binary files, generated
artifacts and cache directories are skipped.

    python tools/check_dashes.py
    python tools/check_dashes.py docs README.md
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
             ".mypy_cache", ".ruff_cache", ".pytest_cache",
             # Skip the organizer task repository when it is checked out beside this one.
             "emmy-hackathon-2026"}
SKIP_SUFFIXES = {".npz", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz",
                 ".tar", ".cubin", ".so", ".o", ".pyc", ".ptx"}
BAD = {0x2014: "em dash", 0x2013: "en dash"}


def iter_files(paths):
    seen = set()
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                yield path
        elif path.is_dir():
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                for name in sorted(filenames):
                    candidate = Path(dirpath) / name
                    if candidate.suffix.lower() in SKIP_SUFFIXES:
                        continue
                    key = candidate.resolve()
                    if key not in seen:
                        seen.add(key)
                        yield candidate


def check_file(path):
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [], str(exc)
    if b"\x00" in data:
        return [], None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [], None
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for col, char in enumerate(line, 1):
            if ord(char) in BAD:
                hits.append((lineno, col, BAD[ord(char)], line))
    return hits, None


def display(path):
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files or directories (default: repository root)")
    args = parser.parse_args(argv)
    targets = [Path(p) for p in args.paths] if args.paths else [ROOT]
    scanned = 0
    failures = 0
    for path in iter_files(targets):
        scanned += 1
        hits, error = check_file(path)
        if error:
            print(f"{display(path)}: cannot read: {error}")
            failures += 1
            continue
        for lineno, col, kind, line in hits:
            snippet = line.replace("\u2014", "[U+2014]").replace("\u2013", "[U+2013]").strip()
            print(f"{display(path)}:{lineno}:{col}: {kind}: {snippet[:120]}")
            failures += 1
    print(f"scanned {scanned} files, {failures} violation(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
