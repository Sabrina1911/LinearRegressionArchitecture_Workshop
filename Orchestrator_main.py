# Orchestrator_main.py
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any  # <-- Any for event callback typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dotenv import load_dotenv
from sqlalchemy import create_engine
import yaml  # YAML config support
import re    # <-- added for MLflow metric key sanitization

# Optional: MLflow experiment tracking
try:
    import mlflow
    _HAVE_MLFLOW = True
except Exception:
    _HAVE_MLFLOW = False

# ---- Project modules ----
from DataPreparation.datapreparation import DataPreparer, DataPrepConfig
from DataExtractionAnalysis.dataextractionanalysis import DataExplorer
from ModelSelection.modelselection import ModelSelector
from ModelTraining.modeltraining import ModelTrainer
from ModelEvaluationValidation.modelevaluationvalidation import ModelEvaluator
from TrainedMLModel.trainedmlmodel import ArtifactStore


# ============================================================
# Logging (console + rotating file handler)
# ============================================================
def setup_logging(out_dir: str | Path = ".") -> None:
    out_dir = Path(out_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # --- prevent duplicate handlers on repeated calls ---
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        log_dir / "run.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ============================================================
# Helpers
# ============================================================
def resolve_db_url(cli_value: str | None) -> str:
    load_dotenv(override=True)
    db_url = cli_value or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("No DB URL provided. Pass --db or set DATABASE_URL in .env")
    return db_url


def build_engine(db_url: str):
    return create_engine(db_url, future=True)


def normalize_thresholds_index(thr: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure thresholds DataFrame has:
      - axis as index
      - 'T_sec' column name (rename 'T' -> 'T_sec' if needed)
    """
    if thr is None or thr.empty:
        return thr

    # If loaded without index, try to set 'axis' as index
    if thr.index.name is None or thr.index.name == "":
        if "axis" in thr.columns:
            thr = thr.set_index("axis")

    # Normalize T column name
    if "T_sec" not in thr.columns and "T" in thr.columns:
        thr = thr.rename(columns={"T": "T_sec"})

    return thr


def _find_spans(mask: np.ndarray, run_len: int) -> List[Tuple[int, int]]:
    """
    Find contiguous (start, end) spans (inclusive, inclusive) in boolean mask
    that are at least `run_len` long.
    """
    if mask.size == 0:
        return []
    spans: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask.astype(bool)):
        if v and start is None:
            start = i
        if not v and start is not None:
            end = i - 1
            if (end - start + 1) >= run_len:
                spans.append((start, end))
            start = None
    # tail
    if start is not None:
        end = len(mask) - 1
        if (end - start + 1) >= run_len:
            spans.append((start, end))
    return spans


# MLflow metric key sanitizer (prevents '#' and other disallowed chars)
def _mlflow_key_safe(name: str) -> str:
    # Allowed: letters, digits, underscore, dash, dot, space, slash
    return re.sub(r"[^0-9A-Za-z_\-./ ]+", "_", name)


# Observer callback for event spans (can later send webhooks)
def _on_event_log(row: Dict[str, Any]) -> None:
    logging.getLogger("events").info(
        f"EVENT axis={row.get('axis')} label={row.get('label')} "
        f"start={row.get('start_time')} end={row.get('end_time')}"
    )
    # Example webhook (disabled):
    # import requests, os
    # url = os.getenv("EVENT_WEBHOOK_URL")
    # if url:
    #     try: requests.post(url, json=row, timeout=2)
    #     except Exception: pass


# ============================================================
# Modes
# ============================================================
def run_train(args):
    log = logging.getLogger("train")
    db_url = resolve_db_url(args.db)
    eng = build_engine(db_url)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: upload TRAIN CSV into DB table (parity with spec)
    if args.train_csv:
        log.info(f"Uploading TRAIN CSV '{args.train_csv}' to table '{args.train_table}'")
        prep = DataPreparer(eng, DataPrepConfig(train_table=args.train_table))
        prep.upload_csv_to_db(args.train_csv, args.train_table)

    # --- Load TRAIN (DB) & TEST (CSV) ---
    expl = DataExplorer(eng)
    train = expl.load_train_from_db(args.train_table)
    if not args.test_csv:
        raise ValueError("test_csv is required (provide via --test_csv or pipeline.yaml).")
    test = expl.load_test_from_csv(args.test_csv)

    # --- Axes selection & alignment ---
    selector = ModelSelector()
    axes_train_8 = selector.pick_axes(train, top_k=8)  # or your policy
    axes_eval = axes_train_8  # keep same set for evaluation
    test_aligned = selector.align_test_to_train_stats(test, train, axes_eval)

    # --- Ensure TRAIN has t_sec (trainer requires it) ---
    if "t_sec" not in train.columns:
        _ev_tmp = ModelEvaluator()
        train = train.copy()
        train["t_sec"] = _ev_tmp._make_time_feature(train)["t_sec"]

    # --- Train per-axis models ---
    trainer = ModelTrainer(random_state=args.random_state)
    models = trainer.fit_models(train, axes_train_8)

    # --- Residuals & thresholds from TRAIN ---
    evaluator = ModelEvaluator()
    train_res = evaluator.compute_residuals(train, models, axes_train_8)

    thresholds = evaluator.build_thresholds_from_residuals(
        train_res,
        cadence_s=args.cadence_s,
        options={"strategy": "IQR", "iqr_k": args.iqr_k, "t_mult": args.t_mult},
    )

    # Persist artifacts (thresholds with index)
    store = ArtifactStore(root=out_dir)
    store.save_models(models)
    store.save_thresholds(thresholds)  # ensure ArtifactStore saves index=True

    # --- Detect events on aligned TEST (matches notebook) ---
    events = evaluator.detect_events(
        test_aligned, models, thresholds, axes_eval, cadence_s=args.cadence_s, on_event=_on_event_log
    )

    # --- Plots: show correct shaded overlay grid (like notebook) ---
    if args.plot in ("overlays", "both"):
        fig = evaluator.overlay_grid(test_aligned, models, events, axes_eval[:8])
        plt.show()

    # --- Optional EDA overlay (if 'both') ---
    if args.plot in ("eda", "both"):
        fig2 = evaluator.overlay_distributions(train, test_aligned, axes_eval[:8])
        plt.show()

    # --- Final PDF summary: grid + decluttered summary ---
    pdf_path = out_dir / "artifacts" / "report.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        figA = evaluator.overlay_grid(test_aligned, models, events, axes_eval[:8])
        pdf.savefig(figA)
        plt.close(figA)

        figB = evaluator.summary_dashboard(
            test_aligned,
            events,
            axes_eval,
            show_topk=4,
            smooth_win=5,
            decimate=3,
            merge_gap_sec=6.0,
            robust_ylim=True,
        )
        pdf.savefig(figB)
        plt.close(figB)

    # --- MLflow logging (optional) ---
    if _HAVE_MLFLOW:
        with mlflow.start_run(run_name=f"train-{args.train_table}"):
            mlflow.log_params({
                "iqr_k": args.iqr_k,
                "t_mult": args.t_mult,
                "cadence_s": args.cadence_s,
                "mode": "train",
                "axes": ",".join(axes_eval[:8]),
            })
            # Per-axis MAE on TRAIN using residuals
            mae_by_axis = {
                ax: float(np.nanmean(np.abs(train_res[ax])))
                for ax in axes_eval if ax in train_res.columns
            }
            for ax, mae in mae_by_axis.items():
                # sanitize metric key to avoid '#' etc.
                mlflow.log_metric(f"mae_train_{_mlflow_key_safe(ax)}", mae)

            art_dir = out_dir / "artifacts"
            pdf_p = art_dir / "report.pdf"
            thr_p = art_dir / "thresholds.csv"
            if pdf_p.exists():
                mlflow.log_artifact(str(pdf_p))
            if thr_p.exists():
                mlflow.log_artifact(str(thr_p))

    log.info(f"Saved PDF summary → {pdf_path}")


def run_evaluate(args):
    log = logging.getLogger("evaluate")
    db_url = resolve_db_url(args.db)
    eng = build_engine(db_url)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load TRAIN from DB and TEST from CSV
    expl = DataExplorer(eng)
    train = expl.load_train_from_db(args.train_table)
    if not args.test_csv:
        raise ValueError("test_csv is required (provide via --test_csv or pipeline.yaml).")
    test = expl.load_test_from_csv(args.test_csv)

    # Load models & thresholds
    store = ArtifactStore(root=out_dir)
    models = store.load_models()

    thr = store.load_thresholds()
    thr = normalize_thresholds_index(thr)  # rename T->T_sec & ensure axis index
    assert {"MinC", "MaxC", "T_sec"}.issubset(set(thr.columns)), \
        "Thresholds missing required columns"

    selector = ModelSelector()
    axes_eval = selector.pick_axes(train, top_k=8)

    # **Align TEST to TRAIN** (parity with train + notebook)
    test_aligned = selector.align_test_to_train_stats(test, train, axes_eval)

    # --- Drift check (post-alignment) ---
    rep = selector.similarity_report(train, test_aligned, axes_eval)
    bad_mean = rep.loc[rep["Δμ_%"].abs() > 5.0, ["axis", "Δμ_%"]]
    bad_std  = rep.loc[(rep["σ_ratio"] < 0.9) | (rep["σ_ratio"] > 1.1), ["axis", "σ_ratio"]]
    if not bad_mean.empty or not bad_std.empty:
        logging.getLogger("evaluate").warning(
            "Drift detected vs TRAIN. "
            f"Mean drift >5% axes: {bad_mean.to_dict(orient='records')} ; "
            f"σ_ratio outside [0.9,1.1] axes: {bad_std.to_dict(orient='records')}"
        )
    drift_path = out_dir / "artifacts" / "drift_similarity_report.csv"
    rep.to_csv(drift_path, index=False)

    evaluator = ModelEvaluator()
    events = evaluator.detect_events(
        test_aligned, models, thr, axes_eval, cadence_s=args.cadence_s, on_event=_on_event_log
    )

    # --- MLflow logging (optional) ---
    if _HAVE_MLFLOW:
        with mlflow.start_run(run_name=f"eval-{args.train_table}"):
            mlflow.log_params({
                "iqr_k": args.iqr_k,
                "t_mult": args.t_mult,
                "cadence_s": args.cadence_s,
                "mode": "evaluate",
                "axes": ",".join(axes_eval[:8]),
            })
            # Compute MAE on EVALUATE set
            Xte = evaluator._make_time_feature(test_aligned)
            mae_by_axis = {}
            for ax in axes_eval:
                if ax in test_aligned.columns and ax in models:
                    y_true = np.asarray(test_aligned[ax], dtype=float)
                    y_pred = np.asarray(models[ax].predict(Xte), dtype=float)
                    mae_by_axis[ax] = float(np.nanmean(np.abs(y_true - y_pred)))
            for ax, mae in mae_by_axis.items():
                # sanitize metric key to avoid '#' etc.
                mlflow.log_metric(f"mae_eval_{_mlflow_key_safe(ax)}", mae)

            art_dir = out_dir / "artifacts"
            pdf_p = art_dir / "report.pdf"
            thr_p = art_dir / "thresholds.csv"
            if pdf_p.exists():
                mlflow.log_artifact(str(pdf_p))
            if thr_p.exists():
                mlflow.log_artifact(str(thr_p))
            if drift_path.exists():
                mlflow.log_artifact(str(drift_path))

    if args.plot in ("overlays", "both"):
        fig = evaluator.overlay_grid(test_aligned, models, events, axes_eval[:8])
        plt.show()

    if args.plot in ("eda", "both"):
        fig2 = evaluator.overlay_distributions(train, test_aligned, axes_eval[:8])
        plt.show()

    log.info("Evaluation complete.")


def run_live(args):
    """
    Example live mode that:
      - loads latest models/thresholds
      - streams/polls data via DataExplorer (pseudo)
      - plots rolling predictions with two-sided ALERT/ERROR spans
    """
    log = logging.getLogger("live")
    db_url = resolve_db_url(args.db)
    eng = build_engine(db_url)

    out_dir = Path(args.out)
    store = ArtifactStore(root=out_dir)
    models = store.load_models()

    thr = store.load_thresholds()
    thr = normalize_thresholds_index(thr)
    assert {"MinC", "MaxC", "T_sec"}.issubset(set(thr.columns)), \
        "Thresholds missing required columns"

    selector = ModelSelector()
    evaluator = ModelEvaluator()

    # Pseudo: acquire a live/test frame (replace with your streaming polling)
    expl = DataExplorer(eng)
    if not args.test_csv:
        raise ValueError("test_csv is required (provide via --test_csv or pipeline.yaml).")
    live_df = expl.load_test_from_csv(args.test_csv)  # stand-in for stream

    axes = list(thr.index) if thr is not None and not thr.empty else selector.pick_axes(live_df, top_k=8)

    # Build residuals vs time for live_df
    preds = {}
    resids = {}
    for ax in axes:
        m = models[ax]
        y = live_df[ax].to_numpy()
        X = evaluator._make_time_feature(live_df)  # or however you built X in train
        yhat = m.predict(X)
        preds[ax] = yhat
        resids[ax] = (y - yhat)

    # Two-sided spans for ALERT/ERROR using dwell T_sec
    t = evaluator._coerce_time(live_df)  # ensure datetime index/series
    fig, axs = plt.subplots(
        nrows=min(4, len(axes)), ncols=2, figsize=(14, 8), squeeze=False, constrained_layout=True  # <-- fix
    )
    axs = axs.ravel()

    for i, ax in enumerate(axes[: len(axs)]):
        series_res = np.asarray(resids[ax], dtype=float)

        # thresholds per axis
        row = thr.loc[ax]
        minC = float(row["MinC"])
        maxC = float(row["MaxC"])
        T_sec = float(row["T_sec"])
        cadence = float(args.cadence_s)
        nT = max(1, int(round(T_sec / max(1e-9, cadence))))

        # Two-sided masks
        alert_mask = (series_res >= minC) | (series_res <= -minC)
        error_mask = (series_res >= maxC) | (series_res <= -maxC)

        alert_spans = _find_spans(alert_mask, nT)
        error_spans = _find_spans(error_mask, nT)

        axp = axs[i]
        axp.plot(t, live_df[ax].values, label=f"{ax} (actual)")
        axp.plot(t, preds[ax], linestyle="--", label=f"{ax} (pred)")

        # Shade spans
        for s, e in alert_spans:
            axp.axvspan(t.iloc[s], t.iloc[e], alpha=0.15, label="ALERT", color=None)
        for s, e in error_spans:
            axp.axvspan(t.iloc[s], t.iloc[e], alpha=0.25, label="ERROR", color=None)

        axp.set_title(ax)
        axp.legend(loc="upper left")

    # removed plt.tight_layout(); constrained_layout=True handles spacing
    plt.show()
    log.info("Live plotting complete.")


# ============================================================
# CLI (enhanced: subcommands + legacy --mode support)
# ============================================================
def build_parser_with_subcommands() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrator for Lab 1 pipeline")
    parser.add_argument("--config", help="Optional: path to pipeline.yaml for default args")  # YAML support
    parser.add_argument("--out", default=".", help="Output directory root")
    parser.add_argument("--db", help="SQLAlchemy DB URL (overrides .env DATABASE_URL)")

    subparsers = parser.add_subparsers(dest="command", required=False)

    # ---------- train ----------
    p_tr = subparsers.add_parser("train", help="Train models and build thresholds")
    p_tr.add_argument("--train_table", default="readings_fact_csv", help="TRAIN table name in DB")
    p_tr.add_argument("--test_csv", required=True, help="Path to TEST CSV file")
    p_tr.add_argument("--train_csv", help="Optional: path to TRAIN CSV (upload to --train_table)")
    p_tr.add_argument("--cadence_s", type=float, default=1.0, help="Sampling cadence (seconds)")
    p_tr.add_argument("--iqr_k", type=float, default=1.5, help="IQR multiplier for MinC/MaxC")
    p_tr.add_argument("--t_mult", type=float, default=3.0, help="Multiplier to convert cadence→T_sec")
    p_tr.add_argument("--plot", choices=["none", "overlays", "eda", "both"], default="overlays")
    p_tr.add_argument("--random_state", type=int, default=42)

    # ---------- evaluate ----------
    p_ev = subparsers.add_parser("evaluate", help="Evaluate on TEST with overlays and reports")
    p_ev.add_argument("--train_table", default="readings_fact_csv", help="TRAIN table name in DB")
    p_ev.add_argument("--test_csv", required=True, help="Path to TEST CSV file")
    p_ev.add_argument("--cadence_s", type=float, default=1.0, help="Sampling cadence (seconds)")
    p_ev.add_argument("--iqr_k", type=float, default=1.5, help="IQR multiplier for MinC/MaxC")
    p_ev.add_argument("--t_mult", type=float, default=3.0, help="Multiplier to convert cadence→T_sec")
    p_ev.add_argument("--plot", choices=["none", "overlays", "eda", "both"], default="overlays")

    # ---------- live ----------
    p_lv = subparsers.add_parser("live", help="Simple live preview (pseudo-stream from CSV)")
    p_lv.add_argument("--train_table", default="readings_fact_csv", help="TRAIN table name in DB")
    p_lv.add_argument("--test_csv", required=True, help="Path to TEST CSV file")
    p_lv.add_argument("--cadence_s", type=float, default=1.0, help="Sampling cadence (seconds)")
    p_lv.add_argument("--plot", choices=["none", "overlays", "eda", "both"], default="overlays")

    # ---------- playback (Cell 19.0-style) ----------
    pb = subparsers.add_parser("playback", help="Live sliding-window visualization (Cell 19.0 style)")
    pb.add_argument("--meta", type=str, default="data/metadata_wilk_clean.csv", help="Path to TEST metadata CSV")
    pb.add_argument("--artifacts", type=str, default="artifacts", help="Models/thresholds directory")
    pb.add_argument("--tick", type=int, default=2, help="Seconds per row (TICK_SECONDS)")
    pb.add_argument("--window", type=int, default=45, help="Rows per sliding window (WINDOW_ROWS)")
    pb.add_argument("--stacked", type=int, default=1, help="1=stacked area, 0=lines")
    pb.add_argument("--topk", type=int, default=8, help="How many axes to visualize")
    pb.add_argument("--train_csv", type=str, default="", help="Optional TRAIN CSV for axis picking/std")

    # ------- legacy compatibility flags (no subcommand used) -------
    parser.add_argument("--mode", choices=["train", "evaluate", "live"], help="LEGACY: use subcommands instead")
    parser.add_argument("--train_table", default="readings_fact_csv", help=argparse.SUPPRESS)
    parser.add_argument("--test_csv", help=argparse.SUPPRESS)
    parser.add_argument("--train_csv", help=argparse.SUPPRESS)
    parser.add_argument("--cadence_s", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--iqr_k", type=float, default=1.5, help=argparse.SUPPRESS)
    parser.add_argument("--t_mult", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--plot", choices=["none", "overlays", "eda", "both"], default="overlays", help=argparse.SUPPRESS)
    parser.add_argument("--random_state", type=int, default=42, help=argparse.SUPPRESS)

    return parser


def merge_yaml_defaults(args: argparse.Namespace) -> argparse.Namespace:
    # --- YAML config merge (YAML defaults < CLI overrides) ---
    if getattr(args, "config", None):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        for k, v in cfg.items():
            if getattr(args, k, None) in (None, "", []):
                setattr(args, k, v)
    return args


def dispatch(args: argparse.Namespace) -> None:
    """
    Dispatches to subcommands; if none provided, falls back to legacy --mode.
    """
    setup_logging(getattr(args, "out", "."))

    # --- Subcommand path ---
    if getattr(args, "command", None):
        logging.getLogger().info(f"Subcommand: {args.command}")
        if args.command == "train":
            run_train(args)
            return
        if args.command == "evaluate":
            run_evaluate(args)
            return
        if args.command == "live":
            run_live(args)
            return
        if args.command == "playback":
            # Reuse the playback script
            from scripts.playback import main as playback_main
            import sys as _sys

            argv = [
                "--meta", args.meta,
                "--artifacts", args.artifacts,
                "--tick", str(args.tick),
                "--window", str(args.window),
                "--stacked", str(args.stacked),
                "--topk", str(args.topk),
            ]
            if args.train_csv:
                argv += ["--train_csv", args.train_csv]

            # playback.main() parses its own argparse; emulate sys.argv
            _sys.argv = ["playback.py"] + argv
            playback_main()
            return

        raise ValueError(f"Unknown subcommand: {args.command}")

    # --- Legacy path (no subcommand specified) ---
    logging.getLogger().info("No subcommand provided; using legacy --mode path.")
    mode = getattr(args, "mode", None) or "train"
    try:
        if mode == "train":
            run_train(args)
        elif mode == "evaluate":
            run_evaluate(args)
        elif mode == "live":
            run_live(args)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as e:
        logging.getLogger("main").exception(f"Fatal error: {e}")
        raise


def main():
    parser = build_parser_with_subcommands()
    args = parser.parse_args()
    args = merge_yaml_defaults(args)
    dispatch(args)


if __name__ == "__main__":
    main()
