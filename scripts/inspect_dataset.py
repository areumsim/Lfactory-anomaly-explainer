"""
데이터셋 인스펙션 스크립트

예)
  python scripts/inspect_dataset.py --dataset SKAB --data-root /path/to/ROOT --split test

출력:
  - head(5) 미리보기 (pandas가 있으면 DataFrame, 없으면 튜플 리스트)
  - label value_counts 유사 통계
  - meta 요약
"""
from __future__ import annotations

import argparse
from typing import Any

from experiments.data.data_router import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect a single dataset file via common loader")
    p.add_argument("--dataset", required=True, choices=["SKAB", "SMD", "AIHub71802"], help="Dataset name")
    p.add_argument("--data-root", required=True, help="Dataset root path")
    p.add_argument("--split", default="test")
    p.add_argument("--label-scheme", default="binary", choices=["binary", "risk4"])
    p.add_argument("--sample-limit", type=int, default=0)
    p.add_argument("--file-index", type=int, default=-1)
    p.add_argument("--min-length", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = load_dataset(
        name=args.dataset,
        data_root=args.data_root,
        split=args.split,
        label_scheme=args.label_scheme,
        sample_limit=(args.sample_limit if args.sample_limit > 0 else None),
        file_index=(args.file_index if args.file_index >= 0 else None),
        min_length=max(0, args.min_length),
    )
    print("\n[meta]", d.get("meta", {}))


if __name__ == "__main__":
    main()

