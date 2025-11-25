#!/usr/bin/env python3
"""Organize runs/* into readable, standardized folders and add REPORT.md.

Pattern: runs/<dataset>_<UTCYYYYMMDD_HHMMSS>_<run_id>_<detector>/
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Dict


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "unknown"


def _render_report(result: Dict, out_dir: str) -> str:
    meta = result.get("meta", {})
    run = result.get("run", {})
    det = result.get("detector", {})
    m = result.get("metrics", {})
    dec = result.get("decision", {}) or {}
    lines = []
    lines.append(f"# 실행 보고서\n")
    lines.append(f"- 데이터셋: {meta.get('dataset', 'unknown')}")
    lines.append(f"- 실행 시각(UTC): {run.get('start_ts', '')}")
    lines.append(f"- Run ID: {run.get('run_id', '')}")
    lines.append(f"- Detector: {det.get('method','')}\n")
    lines.append("## 메트릭")
    lines.append(
        f"- Precision {m.get('precision', 0):.3f}, Recall {m.get('recall', 0):.3f}, F1 {m.get('f1', 0):.3f}, Accuracy {m.get('accuracy', 0):.3f}"
    )
    lines.append(f"- AUC-ROC {m.get('auc_roc', 0):.3f}, AUC-PR {m.get('auc_pr', 0):.3f}, ECE {m.get('ece', 0):.3f}\n")
    lines.append("## 비용 민감 임계")
    if dec:
        lines.append(
            f"- Fixed {dec.get('fixed_expected_cost', 0):.3f} → Optimal {dec.get('optimal_expected_cost', 0):.3f} (Gain {dec.get('expected_cost_gain', 0):.3f})"
        )
        lines.append(
            f"- Thresholds: fixed {dec.get('fixed_threshold', 0):.6g}, optimal {dec.get('optimal_threshold', 0):.6g}\n"
        )
    else:
        lines.append("- (미사용)\n")
    lines.append("## 아티팩트")
    lines.append(f"- 예측 CSV: {os.path.join(out_dir, 'preds.csv')}")
    lines.append(f"- 플롯 디렉토리: {os.path.join(out_dir, 'plots')}\n")
    return "\n".join(lines)


def _target_name(result: Dict) -> str:
    dataset = _slug(result.get("meta", {}).get("dataset", "unknown"))
    ts = result.get("run", {}).get("start_ts", "").replace(":", "").replace("-", "").replace("T", "_")
    ts_compact = ts[:13] if ts else "unknown"
    run_id = _slug(result.get("run", {}).get("run_id", "run"))
    det = _slug(result.get("detector", {}).get("method", "detector"))
    return f"{dataset}_{ts_compact}_{run_id}_{det}"


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    runs_dir = os.path.join(root, "runs")
    if not os.path.isdir(runs_dir):
        print("runs directory not found")
        return
    for name in sorted(os.listdir(runs_dir)):
        src = os.path.join(runs_dir, name)
        if not os.path.isdir(src):
            continue
        run_json = os.path.join(src, "run.json")
        if not os.path.exists(run_json):
            continue
        try:
            with open(run_json, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            continue
        target_base = _target_name(result)
        if name == target_base:
            report_path = os.path.join(src, "REPORT.md")
            if not os.path.exists(report_path):
                with open(report_path, "w", encoding="utf-8") as rf:
                    rf.write(_render_report(result, out_dir=src))
            continue
        dst = os.path.join(runs_dir, target_base)
        suffix = 1
        while os.path.exists(dst):
            dst = os.path.join(runs_dir, f"{target_base}_{suffix}")
            suffix += 1
        shutil.move(src, dst)
        try:
            with open(os.path.join(dst, "REPORT.md"), "w", encoding="utf-8") as rf:
                rf.write(_render_report(result, out_dir=dst))
        except Exception:
            pass
        print(f"moved: {name} -> {os.path.basename(dst)}")


if __name__ == "__main__":
    main()

