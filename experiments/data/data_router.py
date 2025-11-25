from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import loader_skab
from . import loader_smd
from . import loader_aihub_71802


def _load_yaml_simple(path: str) -> Dict[str, Any]:
    # 경량 YAML 로더: PyYAML 우선, 실패 시 1단계 키만 파싱(예: root)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"datasets cfg not found: {path}")
    try:
        import yaml  # type: ignore

        with open(p, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        # minimal parser for `root:` and first-level keys only
        root: Dict[str, Any] = {}
        with open(p, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line and not line.startswith("-"):
                    k, v = line.split(":", 1)
                    k = k.strip(); v = v.strip().strip("\"'")
                    root[k] = v
        return root


def resolve_data_root(dataset: str, datasets_cfg: str) -> Optional[str]:
    # datasets.yaml 기반 루트 자동 해석
    # - root가 있으면 ${root} 치환
    # - datasets:<name>.path 우선, 없으면 root 반환
    # - 실패 시 None 반환
    """Resolve a data root for the given dataset using datasets.yaml.

    Rules:
    - If `root` is present, expand ${root} in dataset paths.
    - For known datasets (SKAB/SMD/AIHub71802), prefer `datasets:<name>.path` if present; else return `root`.
    Returns an absolute path or None if cannot resolve.
    """
    try:
        cfg = _load_yaml_simple(datasets_cfg)
    except Exception as e:
        print(f"[WARN] Failed to read datasets cfg {datasets_cfg}: {e}")
        return None
    ds = dataset.strip()
    base = cfg.get("root") or cfg.get("base") or ""
    def expand(p: str) -> str:
        return os.path.abspath(os.path.expanduser(str(p).replace("${root}", str(base))))
    datasets_map = cfg.get("datasets") if isinstance(cfg.get("datasets"), dict) else None
    if datasets_map:
        entry = datasets_map.get(ds) or datasets_map.get(ds.upper()) or datasets_map.get(ds.lower())
        if isinstance(entry, dict) and "path" in entry:
            return expand(entry["path"])
    if base:
        return expand(base)
    return None


def load_timeseries(
    name: str,
    data_root: str,
    split: str = "test",
    label_scheme: str = "binary",
    sample_limit: int | None = None,
    file_index: int | None = None,
    min_length: int = 0,
) -> Dict[str, Any]:
    """Load a single time-series + labels for the given dataset.

    Returns a dict compatible with the baseline pipeline:
      {"series": List[float], "labels": List[int], "meta": dict}
    """
    ds = name.strip().lower()
    root = os.path.abspath(os.path.expanduser(data_root)) if data_root else ""
    if not root or not os.path.exists(root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    if ds == "skab":
        return loader_skab.load_one_timeseries(
            root, split=split, sample_limit=sample_limit, file_index=file_index, min_length=min_length
        )
    if ds == "smd":
        return loader_smd.load_one_timeseries(
            root, split=split, sample_limit=sample_limit, file_index=file_index, min_length=min_length
        )
    if ds in ("aihub71802", "aihub_71802", "aihub"):
        return loader_aihub_71802.load_one_timeseries(
            root, split=split, label_scheme=label_scheme, sample_limit=sample_limit, file_index=file_index, min_length=min_length
        )

    raise ValueError(f"Unsupported dataset name: {name}")


def load_dataset(
    name: str,
    data_root: str,
    split: str = "test",
    label_scheme: str = "binary",
    sample_limit: int | None = None,
    file_index: int | None = None,
    min_length: int = 0,
) -> Dict[str, Any]:
    """공통 인터페이스: DataFrame/ndarray와 메타데이터 반환

    Returns: {"data": DataFrame or dict of lists, "label": list/ndarray, "meta": dict}
    로그: DataFrame.head() 및 label value_counts 유사 통계 출력
    """
    ts = load_timeseries(
        name=name,
        data_root=data_root,
        split=split,
        label_scheme=label_scheme,
        sample_limit=sample_limit,
        file_index=file_index,
        min_length=min_length,
    )
    series: List[float] = list(float(v) for v in ts["series"])  # type: ignore
    labels: List[int] = list(int(v) for v in ts["labels"])  # type: ignore
    meta: Dict[str, Any] = dict(ts.get("meta", {}))
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame({
            "timestamp": list(range(len(series))),
            "value": series,
            "label": labels,
        })
        print(df.head(5).to_string(index=False))
        vc = df["label"].value_counts().to_dict()
        print(f"[loader] label counts: {vc}")
        data_obj: Any = df
    except Exception:
        data_obj = {"timestamp": list(range(len(series))), "value": series, "label": labels}
        print(f"[loader] head: {[(data_obj['timestamp'][i], data_obj['value'][i], data_obj['label'][i]) for i in range(min(5, len(series)))]}")
        c0 = sum(1 for y in labels if y == 0)
        c1 = sum(1 for y in labels if y != 0)
        print(f"[loader] label counts: {{0: {c0}, 1: {c1}}}")
    return {"data": data_obj, "label": labels, "meta": meta}
