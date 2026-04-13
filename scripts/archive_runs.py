#!/usr/bin/env python3
"""Archive runs/* into runs_archive/<timestamp>/ and delete unknown ones.

Known run criteria:
- Directory contains run.json that parses successfully
- meta.dataset is one of {synthetic, SKAB, SMD, AIHub71802} (case-insensitive)
- run.start_ts is present (ISO string)

Unknown run folders (deleted):
- Missing or unparsable run.json
- meta.dataset unknown or empty
- start_ts missing
- Folder names containing '_unknown_' token
"""
from __future__ import annotations

import json
import os
import re
import shutil
import datetime as dt
from typing import Tuple


ALLOWED_DATASETS = {"synthetic", "skab", "smd", "aihub71802"}


def is_unknown(folder: str, runs_dir: str) -> Tuple[bool, str]:
    path = os.path.join(runs_dir, folder)
    if not os.path.isdir(path):
        return True, "not_a_dir"
    if "_unknown_" in folder:
        return True, "name_unknown_token"
    rj = os.path.join(path, "run.json")
    if not os.path.exists(rj):
        return True, "missing_run_json"
    try:
        with open(rj, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return True, "unparsable_json"
    meta = obj.get("meta", {})
    run = obj.get("run", {})
    ds = str(meta.get("dataset", "")).strip().lower()
    ts = str(run.get("start_ts", "")).strip()
    if ds not in ALLOWED_DATASETS:
        return True, f"unknown_dataset:{ds}"
    if not ts:
        return True, "missing_start_ts"
    return False, "ok"


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    runs_dir = os.path.join(root, "runs")
    if not os.path.isdir(runs_dir):
        print("runs directory not found")
        return
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_root = os.path.join(root, "runs_archive", stamp)
    os.makedirs(archive_root, exist_ok=True)

    archived = []
    removed = []
    for name in sorted(os.listdir(runs_dir)):
        src = os.path.join(runs_dir, name)
        if not os.path.isdir(src):
            continue
        unknown, reason = is_unknown(name, runs_dir)
        if unknown:
            shutil.rmtree(src, ignore_errors=True)
            removed.append((name, reason))
        else:
            dst = os.path.join(archive_root, name)
            shutil.move(src, dst)
            archived.append(name)

    # summary
    print(f"Archived {len(archived)} runs into {archive_root}")
    if removed:
        print("Removed unknown runs:")
        for n, why in removed:
            print(f"  - {n} ({why})")


if __name__ == "__main__":
    main()

