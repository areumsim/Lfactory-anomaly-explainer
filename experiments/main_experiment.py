"""Minimal experimental CLI: Detect mode (first loop).

Research context:
- Implements the Detect step of the Detect–Explain–Act pipeline using a
  synthetic dataset and a rule-based detector. Results are printed and
  can be redirected to JSON/CSV by the runner. Later loops will add
  logging, visualization, calibration, and decision layers.

Usage examples:
  python -m experiments.main_experiment --dataset synthetic --mode detect
  python -m experiments.main_experiment --dataset synthetic --mode detect \
      --length 3000 --anomaly-rate 0.03 --z-threshold 3.5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict
import datetime
import subprocess
from pathlib import Path

from . import data_loader
from .data import load_timeseries
from . import rule_detector
from . import ml_detector_knn
from . import ml_detector_isolation_forest
from . import ml_detector_lstm_ae
from . import hybrid_detector
from . import spec_cnn
from . import metrics
from . import result_manager
from . import calibration
from . import feature_bank
from . import cost_threshold


def _load_simple_yaml(path: str) -> Dict[str, Any]:
    """Load YAML if PyYAML is available, otherwise parse a minimal subset.

    Supported fallback syntax:
      - key: value (scalars: int/float/bool/str)
      - one-level nested maps via two-space indentation.
    This is intentionally simple to avoid heavy deps.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    # Try PyYAML first
    try:
        import yaml  # type: ignore

        with open(p, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if not isinstance(obj, dict):
            return {}
        return obj  # type: ignore
    except Exception:
        pass
    # Fallback: minimal parser
    root: Dict[str, Any] = {}
    cur: Dict[str, Any] | None = None
    cur_key: str | None = None
    with open(p, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            if not line.startswith(" "):
                # top-level key
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if v == "":
                        cur = {}
                        root[k] = cur
                        cur_key = k
                    else:
                        # scalar at top-level
                        root[k] = _parse_scalar(v)
                        cur = None
                        cur_key = None
                continue
            # nested (two-space indented) under cur
            if cur is not None and cur_key is not None:
                l = line.strip()
                if ":" in l:
                    k, v = l.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    cur[k] = _parse_scalar(v)
    return root


def _parse_scalar(v: str) -> Any:
    # strip quotes
    s = v.strip()
    if s.startswith("#"):
        return ""
    if s.startswith("\"") and s.endswith("\""):
        s = s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
    # booleans
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    # numbers
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except Exception:
        return s


def _apply_config_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """Populate argparse args from config if CLI left them as defaults.

    Rule: CLI non-empty values take precedence; otherwise use config.
    """
    def maybe_set(section: str, key: str, arg_name: str) -> None:
        if section in cfg and isinstance(cfg[section], dict) and key in cfg[section]:
            cur = getattr(args, arg_name)
            val = cfg[section][key]
            # Determine emptiness
            if isinstance(cur, str):
                if not cur:
                    setattr(args, arg_name, val)
            else:
                # numeric/bool: set unconditionally (config as baseline)
                setattr(args, arg_name, val)

    # general
    maybe_set("general", "dataset", "dataset")
    maybe_set("general", "data_root", "data_root")
    maybe_set("general", "split", "split")
    maybe_set("general", "label_scheme", "label_scheme")
    maybe_set("general", "seed", "seed")
    maybe_set("general", "run_id", "run_id")
    # output
    maybe_set("output", "out_dir", "out_dir")
    maybe_set("output", "out_json", "out_json")
    maybe_set("output", "out_csv", "out_csv")
    maybe_set("output", "features_csv", "features_csv")
    maybe_set("output", "plots_dir", "plots_dir")
    # detector
    maybe_set("detector", "type", "detector")
    maybe_set("detector", "z_window", "z_window")
    maybe_set("detector", "z_threshold", "z_threshold")
    maybe_set("detector", "min_std", "min_std")
    maybe_set("detector", "z_robust", "z_robust")
    maybe_set("detector", "mad_eps", "mad_eps")
    maybe_set("detector", "ml_k", "ml_k")
    maybe_set("detector", "ml_quantile", "ml_quantile")
    maybe_set("detector", "hybrid_alpha", "hybrid_alpha")
    maybe_set("detector", "hybrid_quantile", "hybrid_quantile")
    maybe_set("detector", "sc_window", "sc_window")
    maybe_set("detector", "sc_hop", "sc_hop")
    maybe_set("detector", "sc_quantile", "sc_quantile")
    # calibration
    maybe_set("calibration", "method", "calibrate")
    maybe_set("calibration", "ece_bins", "ece_bins")
    maybe_set("calibration", "platt_lr", "platt_lr")
    maybe_set("calibration", "platt_epochs", "platt_epochs")
    maybe_set("calibration", "platt_l2", "platt_l2")
    # decision
    maybe_set("decision", "cost_optimize", "cost_optimize")
    maybe_set("decision", "apply_cost_threshold", "apply_cost_threshold")
    maybe_set("decision", "costs", "costs")
    # loader
    maybe_set("loader", "sample_limit", "sample_limit")
    maybe_set("loader", "file_index", "file_index")
    maybe_set("loader", "min_length", "min_length")
    return args


def _render_run_report(result: Dict[str, Any], detector_name: str, out_csv: str, plots_dir: str) -> str:
    meta = result.get("meta", {})
    run = result.get("run", {})
    det = result.get("detector", {})
    m = result.get("metrics", {})
    dec = result.get("decision", {}) or {}
    lines: list[str] = []
    lines.append(f"# 실행 보고서\n")
    lines.append(f"- 데이터셋: {meta.get('dataset', 'unknown')}")
    lines.append(f"- 실행 시각(UTC): {run.get('start_ts', '')}")
    lines.append(f"- Run ID: {run.get('run_id', '')}")
    method = det.get('method', '')
    lines.append(f"- Detector: {detector_name}{(' (' + method + ')') if method else ''}\n")
    lines.append("## 메트릭")
    lines.append(
        f"- Precision {m.get('precision', 0):.3f}, Recall {m.get('recall', 0):.3f}, F1 {m.get('f1', 0):.3f}, Accuracy {m.get('accuracy', 0):.3f}"
    )
    lines.append(f"- AUC-ROC {m.get('auc_roc', 0):.3f}, AUC-PR {m.get('auc_pr', 0):.3f}, ECE {m.get('ece', 0):.3f}\n")
    ev = result.get("event_metrics") or {}
    if ev:
        lines.append("## 이벤트 타이밍")
        lines.append(
            f"- Events {ev.get('num_events',0):.0f}, Detected {ev.get('detected_events',0):.0f}, "
            f"Delay μ/med {ev.get('mean_detection_delay',0):.3f}/{ev.get('median_detection_delay',0):.3f}, Lead μ/med {ev.get('mean_lead_time',0):.3f}/{ev.get('median_lead_time',0):.3f}\n"
        )
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
    # If thresholded metrics were produced, show A/B comparison succinctly
    th = result.get("thresholded")
    if th and isinstance(th, dict):
        tm = th.get("metrics", {})
        lines.append("\n## 임계 적용(A/B)")
        lines.append(f"- Optimal Thr: {th.get('threshold', 0):.6g}")
        # Compact table for A/B comparison
        lines.append("")
        lines.append("| Metric | Original | Cost-Opt |")
        lines.append("|---|---:|---:|")
        lines.append(f"| F1 | {m.get('f1',0):.3f} | {tm.get('f1',0):.3f} |")
        lines.append(f"| Accuracy | {m.get('accuracy',0):.3f} | {tm.get('accuracy',0):.3f} |")
        if dec:
            lines.append(f"| Expected Cost | {dec.get('fixed_expected_cost',0):.3f} | {dec.get('optimal_expected_cost',0):.3f} |")
        tev = th.get("event_metrics") or {}
        if tev:
            lines.append(f"| Event F1 | {(result.get('event_metrics') or {}).get('event_f1',0):.3f} | {tev.get('event_f1',0):.3f} |")
        lines.append("")

    lines.append("## 아티팩트")
    lines.append(f"- 예측 CSV: {out_csv}")
    lines.append(f"- 플롯 디렉토리: {plots_dir}\n")
    return "\n".join(lines)


def run_detect(args: argparse.Namespace) -> Dict[str, Any]:
    # Run-level metadata (seed, git sha, timestamp, run_id)
    start_ts = datetime.datetime.utcnow().isoformat() + "Z"
    run_id = getattr(args, "run_id", "") or ""
    git_sha = "unknown"
    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass
    # Dataset dispatch: synthetic (built-in) vs external datasets via router
    if args.dataset.lower() == "synthetic":
        data = data_loader.load(
            args.dataset,
            length=args.length,
            anomaly_rate=args.anomaly_rate,
            noise_std=args.noise_std,
            seed=args.seed,
        )
    else:
        if not args.data_root:
            # (주석) --data-root 미지정 시 datasets.yaml로 자동 해석을 시도합니다
            #       실패하면 명시적으로 --data-root를 요구합니다
            # Try datasets.yaml resolver if provided
            try:
                if getattr(args, "datasets_cfg", ""):
                    from .data.data_router import resolve_data_root
                    guess = resolve_data_root(args.dataset, args.datasets_cfg)
                    if guess and os.path.exists(guess):
                        args.data_root = guess
            except Exception:
                pass
            if not args.data_root:
                raise SystemExit("--data-root is required for real datasets (or provide --datasets-cfg)")
        data = load_timeseries(
            name=args.dataset,
            data_root=args.data_root,
            split=args.split,
            label_scheme=args.label_scheme,
            sample_limit=args.sample_limit,
            file_index=(args.file_index if args.file_index >= 0 else None),
            min_length=max(0, args.min_length),
        )

    # Auto organize run directory with dataset + date for clarity
    ds_name = args.dataset
    ts_compact = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rid = getattr(args, "run_id", "") or f"seed{getattr(args, 'seed', 0)}"
    auto_run_dir = args.out_dir.strip() if getattr(args, "out_dir", "") else os.path.join("runs", f"{ds_name}_{ts_compact}_{rid}")

    # Detector selection
    if args.detector == "rule":
        if getattr(args, "z_robust", False):
            det = rule_detector.robust_zscore_detect(
                data["series"],
                window=args.z_window,
                threshold=args.z_threshold,
                min_mad=args.mad_eps,
            )
        else:
            det = rule_detector.zscore_detect(
                data["series"],
                window=args.z_window,
                threshold=args.z_threshold,
                min_std=args.min_std,
            )
    elif args.detector == "ml":
        ml_method = getattr(args, "ml_method", "knn")
        if ml_method == "knn":
            det = ml_detector_knn.detect_knn(
                data["series"],
                k=args.ml_k,
                quantile=args.ml_quantile,
            )
        elif ml_method == "isolation_forest":
            det = ml_detector_isolation_forest.detect_isolation_forest(
                data["series"],
                window_size=getattr(args, "if_window", 50),
                contamination=getattr(args, "if_contamination", 0.1),
                n_estimators=getattr(args, "if_estimators", 100),
                random_state=args.seed,
            )
        elif ml_method == "lstm_ae":
            det = ml_detector_lstm_ae.detect_lstm_ae(
                data["series"],
                sequence_length=getattr(args, "lstm_seq_len", 50),
                latent_dim=getattr(args, "lstm_latent_dim", 32),
                epochs=getattr(args, "lstm_epochs", 50),
                learning_rate=getattr(args, "lstm_lr", 0.001),
                batch_size=getattr(args, "lstm_batch_size", 32),
                quantile=getattr(args, "lstm_quantile", 0.95),
                random_state=args.seed,
            )
        else:
            raise ValueError(f"Unsupported ml_method: {ml_method}")
    elif args.detector == "hybrid":
        det = hybrid_detector.detect_hybrid(
            data["series"],
            alpha=args.hybrid_alpha,
            rule_window=args.z_window,
            rule_threshold=args.z_threshold,
            rule_min_std=args.min_std,
            robust=args.z_robust,
            ml_k=args.ml_k,
            quantile=args.hybrid_quantile,
        )
    elif args.detector == "speccnn":
        det = spec_cnn.detect_speccnn(
            data["series"],
            window=args.sc_window,
            hop=args.sc_hop,
            quantile=args.sc_quantile,
        )
    else:
        raise ValueError(f"Unsupported detector: {args.detector}")

    # Label handling: support risk4 by binarizing (>0 -> 1) for metrics/curves
    labels_raw = list(int(v) for v in data["labels"])  # may be 0/1 or 0..3
    label_scheme = str((data.get("meta") or {}).get("label_scheme", "binary")).lower()
    if label_scheme == "risk4" or any(v not in (0, 1) for v in labels_raw):
        labels_metric = [1 if v > 0 else 0 for v in labels_raw]
        label_policy = "risk4>0->1"
    else:
        labels_metric = labels_raw
        label_policy = "binary"
    m = metrics.binary_metrics(labels_metric, det["preds"])
    # ROC/PR from scores (threshold sweep)
    roc = result_manager.compute_roc(labels_metric, det["scores"])
    pr = result_manager.compute_pr(labels_metric, det["scores"])
    m["auc_roc"] = roc.auc
    m["auc_pr"] = pr.auc

    # Event-based metrics (if any positive labels exist)
    try:
        segs = metrics.segments_from_labels(labels_metric)
        if segs:
            m_event = metrics.event_metrics_from_segments(segs, det["preds"])
        else:
            m_event = None
    except Exception:
        m_event = None

    # Probabilities for calibration metrics/plots
    if args.calibrate == "platt":
        A, B = calibration.fit_platt(det["scores"], data["labels"], lr=args.platt_lr, epochs=args.platt_epochs, l2=args.platt_l2)
        probs = calibration.apply_platt(det["scores"], A, B)
        cal_meta = {"method": "platt", "A": A, "B": B}
    elif args.calibrate == "temperature":
        mu, std, T = calibration.fit_temperature(det["scores"], data["labels"])  # defaults are reasonable
        probs = calibration.apply_temperature(det["scores"], mu, std, T)
        cal_meta = {"method": "temperature", "mu": mu, "std": std, "T": T}
    elif args.calibrate == "isotonic":
        model = calibration.fit_isotonic(det["scores"], data["labels"])
        probs = calibration.apply_isotonic(det["scores"], model)
        cal_meta = {"method": "isotonic", "num_steps": len(model)}
    else:
        probs = calibration.normalize_scores(det["scores"])  # heuristic baseline
        cal_meta = {"method": "normalize_minmax"}
    m["ece"] = calibration.ece(labels_metric, probs, bins=args.ece_bins)

    # Cost-sensitive thresholding (optional)
    decision = None
    if getattr(args, "cost_optimize", False):
        parts = (args.costs.split(",") if args.costs else [])
        if len(parts) == 4:
            costs = tuple(float(x) for x in parts)  # type: ignore
        else:
            costs = (0.0, 1.0, 5.0, 0.0)
        # Fixed threshold: align with detector configuration
        if args.detector == "rule":
            fixed_thr = args.z_threshold
        else:
            # derive quantile threshold from scores using the correct knob per detector
            sc = sorted(float(s) for s in det["scores"])
            if sc:
                if args.detector == "ml":
                    q = float(args.ml_quantile)
                elif args.detector == "hybrid":
                    q = float(args.hybrid_quantile)
                elif args.detector == "speccnn":
                    q = float(args.sc_quantile)
                else:
                    q = 0.99
                pos = min(max(int(q * (len(sc) - 1)), 0), len(sc) - 1)
                fixed_thr = sc[pos]
            else:
                fixed_thr = 0.0
        fixed_cost = cost_threshold.expected_cost_from_scores(labels_metric, det["scores"], fixed_thr, costs)  # type: ignore
        opt_thr, opt_cost = cost_threshold.find_optimal_threshold(labels_metric, det["scores"], costs)  # type: ignore
        decision = {
            "costs": {"c00": costs[0], "c01": costs[1], "c10": costs[2], "c11": costs[3]},
            "fixed_threshold": fixed_thr,
            "fixed_expected_cost": fixed_cost,
            "optimal_threshold": opt_thr,
            "optimal_expected_cost": opt_cost,
            "expected_cost_gain": fixed_cost - opt_cost,
        }

    # Optional: apply cost-optimal threshold to produce alternative predictions/metrics
    thresholded = None
    if decision is not None and getattr(args, "apply_cost_threshold", False):
        opt_thr = float(decision["optimal_threshold"])
        alt_preds = [1 if float(s) >= opt_thr else 0 for s in det["scores"]]
        alt_metrics = metrics.binary_metrics(labels_metric, alt_preds)
        # Reuse ROC/PR AUC (score-based)
        alt_metrics["auc_roc"] = roc.auc
        alt_metrics["auc_pr"] = pr.auc
        # Event metrics for alt preds
        try:
            if segs:
                alt_event = metrics.event_metrics_from_segments(segs, alt_preds)
            else:
                alt_event = None
        except Exception:
            alt_event = None
        thresholded = {"threshold": opt_thr, "metrics": alt_metrics, "event_metrics": alt_event, "note": "Applied optimal threshold"}

    result = {
        "meta": {**(data["meta"] or {}), "label_policy": label_policy},
        "detector": det["params"],
        "metrics": m,
        "event_metrics": m_event,
        "calibration": cal_meta,
        "decision": decision,
        "thresholded": thresholded,
        "run": {
            "run_id": run_id,
            "seed": int(getattr(args, "seed", 0)),
            "git_sha": git_sha,
            "start_ts": start_ts,
        },
    }
    # Optional artifact persistence (auto-run dir defaults)
    plots_dir = args.plots_dir or os.path.join(auto_run_dir, "plots")
    out_json = args.out_json or os.path.join(auto_run_dir, "run.json")
    out_csv = args.out_csv or os.path.join(auto_run_dir, "preds.csv")
    features_csv = args.features_csv or ""

    # Ensure base directory exists
    result_manager.ensure_dir(auto_run_dir)

    if out_csv:
        result_manager.save_predictions_csv(out_csv, data["series"], data["labels"], det["scores"], det["preds"], probs)
        if thresholded is not None:
            alt_csv = os.path.join(auto_run_dir, "preds_cost_opt.csv")
            alt_thr = float(thresholded["threshold"])
            alt_preds = [1 if float(s) >= alt_thr else 0 for s in det["scores"]]
            result_manager.save_predictions_csv(alt_csv, data["series"], data["labels"], det["scores"], alt_preds, probs)
    if features_csv:
        feats = feature_bank.compute_basic(data["series"])
        result_manager.save_features_csv(features_csv, feats)
    if plots_dir:
        result_manager.ensure_dir(plots_dir)
        roc_csv = os.path.join(plots_dir, "roc_curve.csv")
        result_manager.save_roc_csv(roc_csv, roc)
        roc_png = os.path.join(plots_dir, "roc_curve.png")
        result_manager.save_roc_plot(roc_png, roc)
        pr_csv = os.path.join(plots_dir, "pr_curve.csv")
        result_manager.save_pr_csv(pr_csv, pr)
        pr_png = os.path.join(plots_dir, "pr_curve.png")
        result_manager.save_pr_plot(pr_png, pr)
        cal_png = os.path.join(plots_dir, "calibration.png")
        # Calibration plot/ECE bins aligned to args.ece_bins; use binary labels used for metrics
        result_manager.save_calibration_plot(cal_png, labels_metric, probs, bins=args.ece_bins)

    # Save run.json (use auto default if not set)
    if out_json:
        result_manager.save_json(result, out_json)
    try:
        args_snap = os.path.join(auto_run_dir, "args.json")  # 실행 인자 스냅샷(재현성)
        result_manager.save_json({k: getattr(args, k) for k in vars(args)}, args_snap)
        # If a config file was used, copy a snapshot for reproducibility
        if getattr(args, "config", "") and os.path.exists(args.config):
            dst_cfg = os.path.join(auto_run_dir, "config_snapshot.yaml")
            try:
                shutil.copyfile(args.config, dst_cfg)
            except Exception:
                pass
    except Exception:
        pass

    # Auto-generate per-run REPORT.md for quick human consumption
    try:
        report_path = os.path.join(auto_run_dir, "REPORT.md")
        with open(report_path, "w", encoding="utf-8") as rf:
            rf.write(_render_run_report(result, detector_name=args.detector, out_csv=os.path.abspath(out_csv), plots_dir=os.path.abspath(plots_dir)))
    except Exception:
        pass

    # Phase 2: Generate LLM-guided explanation if requested
    if getattr(args, "explain", False):
        try:
            from .explain_rag import RAGExplainer

            # Initialize explainer
            llm_provider = getattr(args, "llm_provider", "") or None
            llm_config_path = getattr(args, "llm_config", "experiments/llm_config.yaml")
            explainer = RAGExplainer(config_path=llm_config_path, llm_provider=llm_provider)

            # Build context from detection results
            m = result.get("metrics", {})
            cal = result.get("calibration", {})
            dec = result.get("decision", {})

            context = {
                "dataset": args.dataset,
                "detector": args.detector,
                "imbalance": data.get("info", {}).get("anomaly_rate", 0.0),
                "snr": 0.0,  # TODO: Compute SNR from data
                "auc_pr": m.get("auc_pr", 0.0),
                "f1": m.get("f1", 0.0),
                "ece": cal.get("ece", 0.0) if cal else 0.0,
                "expected_cost": dec.get("optimal_expected_cost", 0.0) if dec else 0.0,
            }

            # Generate explanations for key questions
            explanations = {}

            # Question 1: Why use calibration?
            if args.calibrate != "none":
                exp1 = explainer.explain(
                    query="Why should I use calibration in anomaly detection?",
                    context=context
                )
                explanations["calibration"] = exp1

            # Question 2: How to set cost matrix?
            if args.cost_optimize:
                exp2 = explainer.explain(
                    query="How should I set the FN/FP cost ratio?",
                    context=context
                )
                explanations["cost_matrix"] = exp2

            # Question 3: General performance explanation
            exp3 = explainer.explain(
                query=f"Why did the {args.detector} detector achieve F1={m.get('f1', 0):.3f} on {args.dataset}?",
                context=context
            )
            explanations["performance"] = exp3

            # Add to result
            result["explanations"] = {
                "enabled": True,
                "llm_provider": llm_provider or "TF-IDF_only",
                "queries": explanations,
            }

            # Save explanations to markdown file
            try:
                explain_md = os.path.join(auto_run_dir, "EXPLANATIONS.md")
                with open(explain_md, "w", encoding="utf-8") as ef:
                    ef.write("# LLM-Guided Explanations (Phase 2 Prototype)\n\n")
                    ef.write(f"**LLM Provider**: {llm_provider or 'TF-IDF only'}\n\n")
                    ef.write(f"**Detection Context**:\n")
                    for k, v in context.items():
                        ef.write(f"- {k}: {v}\n")
                    ef.write("\n---\n\n")

                    for query_type, exp_result in explanations.items():
                        ef.write(f"## {query_type.replace('_', ' ').title()}\n\n")
                        ef.write(exp_result["explanation"])
                        ef.write("\n\n---\n\n")

                print(f"[Phase 2] Explanations saved to: {explain_md}")
            except Exception as e:
                print(f"[WARN] Failed to save explanations markdown: {e}")

        except ImportError:
            print("[WARN] --explain flag set but explain_rag module not available. Skipping explanation.")
            result["explanations"] = {"enabled": False, "error": "explain_rag module not found"}
        except Exception as e:
            print(f"[WARN] Explanation generation failed: {e}")
            result["explanations"] = {"enabled": False, "error": str(e)}
    else:
        result["explanations"] = {"enabled": False}

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-Guided Local AD – minimal detect loop")
    p.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "SKAB", "SMD", "AIHub71802"],
        help="Dataset name",
    )
    p.add_argument("--config", type=str, default="", help="YAML config path to override defaults (CLI wins if set)")
    p.add_argument("--mode", default="detect", choices=["detect"], help="Execution mode")
    p.add_argument("--detector", choices=["rule", "ml", "hybrid", "speccnn"], default="rule", help="Detector type")

    # Synthetic data params
    p.add_argument("--length", type=int, default=2000)
    p.add_argument("--anomaly-rate", type=float, default=0.02)
    p.add_argument("--noise-std", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # Real dataset params
    p.add_argument("--data-root", type=str, default="", help="Root directory containing datasets (for SKAB/SMD/AIHub71802)")
    p.add_argument("--datasets-cfg", type=str, default="", help="Optional datasets.yaml path to resolve data roots when --data-root is empty")
    p.add_argument("--split", type=str, default="test", help="Split name (e.g., train/test/Validation)")
    p.add_argument("--label-scheme", choices=["binary", "risk4"], default="binary")
    p.add_argument("--sample-limit", type=int, default=0, help="If >0, limit number of rows for quick smoke runs")
    p.add_argument("--file-index", type=int, default=-1, help="If >=0, pick Nth file among candidates (after min_length filter)")
    p.add_argument("--min-length", type=int, default=0, help="If >0, prefer files with at least this many rows")

    # Rule detector params
    p.add_argument("--z-window", type=int, default=50)
    p.add_argument("--z-threshold", type=float, default=3.0)
    p.add_argument("--min-std", type=float, default=1e-3)
    p.add_argument("--z-robust", action="store_true", help="Use robust rolling z-score (median/MAD)")
    p.add_argument("--mad-eps", type=float, default=1e-6, help="MAD lower bound for robust z-score")

    # Output
    p.add_argument("--run-id", type=str, default="", help="Run identifier recorded in outputs and logs")
    p.add_argument("--out-dir", type=str, default="", help="Base directory to save artifacts; default organizes as runs/<dataset>_<date>_<run-id>")
    p.add_argument("--out-json", type=str, default="", help="If set, write results JSON to this path")
    p.add_argument("--out-csv", type=str, default="", help="If set, write per-point predictions CSV to this path")
    p.add_argument("--features-csv", type=str, default="", help="If set, write basic features CSV to this path")
    p.add_argument(
        "--plots-dir",
        type=str,
        default="",
        help="If set, save ROC CSV/PNG and heuristic calibration plot to this directory",
    )
    # Calibration
    p.add_argument("--calibrate", choices=["none", "platt", "isotonic", "temperature"], default="none")
    p.add_argument("--ece-bins", type=int, default=10)
    p.add_argument("--platt-lr", type=float, default=1e-2)
    p.add_argument("--platt-epochs", type=int, default=2000)
    p.add_argument("--platt-l2", type=float, default=1e-3)

    # Decision: cost-sensitive threshold (optional)
    p.add_argument("--cost-optimize", action="store_true", help="If set, compute cost-optimal threshold on scores")
    p.add_argument("--costs", type=str, default="", help="Cost matrix as 'c00,c01,c10,c11' (default 0,1,5,0)")
    p.add_argument("--apply-cost-threshold", action="store_true", help="If set with --cost-optimize, recompute predictions/metrics at optimal threshold and save preds_cost_opt.csv")

    # ML detector params
    p.add_argument("--ml-method", type=str, choices=["knn", "isolation_forest", "lstm_ae"], default="knn", help="ML detector method (knn/isolation_forest/lstm_ae)")
    p.add_argument("--ml-k", type=int, default=10, help="k for kNN value-density detector")
    p.add_argument("--ml-quantile", type=float, default=0.99, help="Quantile threshold for ML detector scores")
    # IsolationForest params
    p.add_argument("--if-window", type=int, default=50, help="Window size for IsolationForest temporal features")
    p.add_argument("--if-contamination", type=float, default=0.1, help="Expected proportion of anomalies for IsolationForest")
    p.add_argument("--if-estimators", type=int, default=100, help="Number of trees in IsolationForest")
    # LSTM Autoencoder params
    p.add_argument("--lstm-seq-len", type=int, default=50, help="Sequence length for LSTM Autoencoder")
    p.add_argument("--lstm-latent-dim", type=int, default=32, help="Latent dimension for LSTM Autoencoder")
    p.add_argument("--lstm-epochs", type=int, default=50, help="Training epochs for LSTM Autoencoder")
    p.add_argument("--lstm-lr", type=float, default=0.001, help="Learning rate for LSTM Autoencoder")
    p.add_argument("--lstm-batch-size", type=int, default=32, help="Batch size for LSTM Autoencoder")
    p.add_argument("--lstm-quantile", type=float, default=0.95, help="Quantile threshold for LSTM Autoencoder")

    # Hybrid detector params
    p.add_argument("--hybrid-alpha", type=float, default=0.5, help="Weight for ML score (0..1)")
    p.add_argument("--hybrid-quantile", type=float, default=0.99, help="Quantile threshold for hybrid scores")

    # SpecCNN-lite params
    p.add_argument("--sc-window", type=int, default=128, help="SpecCNN-lite window size")
    p.add_argument("--sc-hop", type=int, default=16, help="SpecCNN-lite hop size")
    p.add_argument("--sc-quantile", type=float, default=0.99, help="Quantile threshold for SpecCNN-lite scores")

    # Explanation (Phase 2 - RAG-Bayes prototype)
    p.add_argument("--explain", action="store_true", help="Generate LLM-guided explanation for results (Phase 2 prototype)")
    p.add_argument("--llm-provider", type=str, default="", help="LLM provider name (e.g., 'openai_gpt35', 'local_exaone_35_78b'). If empty with --explain, uses TF-IDF retrieval only.")
    p.add_argument("--llm-config", type=str, default="experiments/llm_config.yaml", help="Path to LLM config file (default: experiments/llm_config.yaml)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Apply config overrides if provided
    if getattr(args, "config", ""):
        try:
            cfg = _load_simple_yaml(args.config)
            args = _apply_config_overrides(args, cfg)
        except Exception as e:
            print(f"[WARN] Failed to load config {args.config}: {e}")
    if args.mode == "detect":
        result = run_detect(args)
        if args.out_json:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
