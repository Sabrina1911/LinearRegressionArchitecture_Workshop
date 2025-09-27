# scripts/playback.py
r"""
Cell 19.0-style live playback with stacked overlays.

- Picks axes (top-8 by variance) from TRAIN if available; else from TEST.
- Builds/uses a time feature.
- Loads trained models + thresholds from ArtifactStore.
- Computes residuals on the full TEST once, then plays a sliding window.
- Converts T_sec -> dwell-in-ticks for span finding.
- Draws ALERT (light yellow) / ERROR (light red) shaded spans, stacked area/lines, and a status bar.

Run (PowerShell, from repo root):
    python .\Orchestrator_main.py playback --meta .\data\metadata_wilk_clean.csv --artifacts . --tick 2 --window 45 --stacked 1 --topk 8
    # If your thresholds are in Amps and you want current-based overlays:
    # --overlay_mode current
"""

from __future__ import annotations

import os
import math
import argparse
import time
from collections import deque
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# graceful fallback if IPython isn't installed
try:
    from IPython.display import clear_output  # type: ignore
except Exception:
    def clear_output(wait: bool = False):  # type: ignore
        pass

# --- Project modules ---
from DataExtractionAnalysis.dataextractionanalysis import DataExplorer  # noqa: F401
from DataPreparation.datapreparation import DataPreparer  # noqa: F401
from ModelSelection.modelselection import ModelSelector
from ModelTraining.modeltraining import ModelTrainer  # noqa: F401
from ModelEvaluationValidation.modelevaluationvalidation import ModelEvaluator
from TrainedMLModel.trainedmlmodel import ArtifactStore


# ---------- Helpers ----------
def pick_axes(train: pd.DataFrame | None, test: pd.DataFrame, top_k: int = 8) -> Sequence[str]:
    ms = ModelSelector()
    if train is not None and not train.empty:
        common = [c for c in train.columns if c in test.columns and c.lower().startswith("axis")]
        if not common:
            return ms.pick_axes(test, top_k=top_k)
        return ms.pick_axes(train[common], top_k=top_k)
    return ms.pick_axes(test, top_k=top_k)


def get_time_vector(df: pd.DataFrame) -> np.ndarray:
    try:
        me = ModelEvaluator()
        out = me._make_time_feature(df.copy())  # ensures 't_sec'
        return out["t_sec"].to_numpy(dtype=float)
    except Exception:
        if "Time" in df.columns:
            ts = pd.to_datetime(df["Time"], errors="coerce")
            if ts.notna().any():
                t0 = ts.dropna().iloc[0]
                return (ts.fillna(t0) - t0).dt.total_seconds().to_numpy(dtype=float)
        return np.arange(len(df), dtype=float)


def load_artifacts(artifacts_root: str = ".") -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Pass the PROJECT ROOT as artifacts_root (ArtifactStore appends 'artifacts/...').
    """
    store = ArtifactStore(root=artifacts_root)
    models = store.load_models()
    thresholds = store.load_thresholds()

    if thresholds.index.name != "axis":
        if "axis" in thresholds.columns:
            thresholds = thresholds.set_index("axis")
        thresholds.index.name = "axis"
    if "T_sec" not in thresholds.columns and "T" in thresholds.columns:
        thresholds = thresholds.rename(columns={"T": "T_sec"})
    return models, thresholds


def compute_residuals_full(test: pd.DataFrame, models: Dict[str, object], axes: Sequence[str]) -> pd.DataFrame:
    me = ModelEvaluator()
    res = me.compute_residuals(test.copy(), models, axes)
    for ax in axes:
        if ax in res.columns:
            res[ax] = pd.to_numeric(res[ax], errors="coerce")
    return res


def zscore_residuals_window(
    res_full: pd.DataFrame,
    res_train_std: Dict[str, float],
    win_idx: Sequence[int],
    axes: Sequence[str],
) -> Dict[str, np.ndarray]:
    z: Dict[str, np.ndarray] = {}
    for ax in axes:
        r = pd.to_numeric(res_full[ax], errors="coerce").to_numpy(dtype=float)[win_idx]
        sd = float(res_train_std.get(ax, np.nan))
        z[ax] = r if (not np.isfinite(sd) or sd <= 1e-12) else r / sd
    return z


def find_spans(mask: np.ndarray, run_len: int) -> Sequence[Tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if (j - i + 1) >= run_len:
                spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def overlay_spans(ax, x: np.ndarray, spans: Sequence[Tuple[int, int]], color="orange", alpha=0.35, zorder=5):
    for (a, b) in spans:
        xa = x[a]
        xb = x[b] if b < len(x) else x[-1]
        ax.axvspan(xa, xb, color=color, alpha=alpha, zorder=zorder)


def main():
    p = argparse.ArgumentParser(description="Cell 19.0-style stacked live playback")
    p.add_argument("--meta", type=str, default="data/metadata_wilk_clean.csv", help="Path to TEST metadata CSV")
    p.add_argument("--artifacts", type=str, default=".", help="Artifacts ROOT (project root)")
    p.add_argument("--tick", type=int, default=2, help="Seconds per row (TICK_SECONDS)")
    p.add_argument("--window", type=int, default=45, help="Rows per sliding window (WINDOW_ROWS)")
    p.add_argument("--stacked", type=int, default=1, help="1=stacked area on, 0=off")
    p.add_argument("--topk", type=int, default=8, help="How many axes to visualize")
    p.add_argument("--train_csv", type=str, default="", help="Optional TRAIN CSV (axis picking/std)")
    # NEW: overlay mode selector
    p.add_argument(
        "--overlay_mode",
        choices=["z", "current"],
        default="z",
        help="Use 'z' (|z|>=2/3) or 'current' (compare to MinC/MaxC in thresholds) for ALERT/ERROR.",
    )
    args = p.parse_args()

    # ---- Load TEST
    test = pd.read_csv(args.meta)
    print(f"Loaded TEST metadata: {test.shape} from {args.meta}")

    # Optional TRAIN for axis picking and residual std
    train = None
    if args.train_csv and os.path.exists(args.train_csv):
        try:
            train = pd.read_csv(args.train_csv)
            print(f"Loaded TRAIN CSV: {train.shape} from {args.train_csv}")
        except Exception as e:
            print(f"Warning: could not read TRAIN CSV ({e}). Proceeding without it.")

    # ---- Pick axes
    axes = pick_axes(train, test, top_k=args.topk)
    if not axes:
        raise RuntimeError("No axes found to plot (looking for columns that start with 'Axis').")
    print(f"Axes selected: {axes}")

    # ---- Time vector (seconds)
    _t_test = get_time_vector(test)  # kept for parity / future use

    # ---- Load artifacts (models + thresholds)
    models, thresholds = load_artifacts(args.artifacts)

    # Restrict thresholds to selected axes, drop missing
    thr = thresholds.reindex(axes).dropna(subset=["MinC", "MaxC"])
    missing_thr = [ax for ax in axes if ax not in thr.index]
    if missing_thr:
        print(f"Warning: missing thresholds for axes: {missing_thr}. They will be skipped in overlay logic.")

    # ---- Compute residuals on full TEST
    res_full = compute_residuals_full(test, models, axes)

    # ---- TRAIN residual std per axis (for z-scores)
    res_train_std: Dict[str, float] = {}
    if "resid_std" in thr.columns:
        res_train_std = {ax: float(thr.loc[ax, "resid_std"]) for ax in thr.index}
    for ax in axes:
        if ax not in res_train_std or not np.isfinite(res_train_std[ax]):
            res_train_std[ax] = float(pd.to_numeric(res_full[ax], errors="coerce").std(ddof=1))

    # ---- Live sliding window (non-blocking UI)
    TICK_SECONDS = max(1, int(args.tick))
    WINDOW_ROWS = max(3, int(args.window))
    STACKED = bool(args.stacked)

    buffer_idx = deque(maxlen=WINDOW_ROWS)
    i = 0

    plt.ion()  # interactive mode (non-blocking)
    fig, axp = plt.subplots(figsize=(12, 5))

    # fixed axis color cycle (8 strong colors)
    axis_palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#ff00ff",  # magenta
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # grey
    ]

    try:
        while i < len(test):
            buffer_idx.append(i)

            # Build current window
            win_idx = list(buffer_idx)
            x = np.arange(len(win_idx), dtype=int)

            # Per-window z-scores
            z_win = zscore_residuals_window(res_full, res_train_std, win_idx, axes)

            # ---- Plot (non-blocking) ----
            axp.clear()

            # lighter pastel overlays (on top of the data)
            overlay_kwargs = dict(alpha=0.15, zorder=5)

            # --- Overlays per axis (selectable mode) ---
            window_alerts = 0
            window_errors = 0

            for ax in axes:
                if ax not in thr.index:
                    continue

                th = thr.loc[ax]
                T_sec = float(th.get("T_sec", th.get("T", 4.0)))
                nT = max(1, int(math.ceil(T_sec / max(1e-6, TICK_SECONDS))))

                if args.overlay_mode == "current":
                    # RAW current vs MinC / MaxC (Amps)
                    y = pd.to_numeric(test[ax], errors="coerce").fillna(0.0).to_numpy(dtype=float)[win_idx]
                    minC = float(th.get("MinC", np.nan))
                    maxC = float(th.get("MaxC", np.nan))
                    alert_mask = np.isfinite(minC) & (y >= minC)
                    error_mask = np.isfinite(maxC) & (y >= maxC)
                else:
                    # DEFAULT z-mode (robust)
                    z = z_win[ax]  # already sliced to window
                    alert_mask = np.abs(z) >= 2.0   # ALERT at |z| >= 2
                    error_mask = np.abs(z) >= 3.0   # ERROR  at |z| >= 3

                alert_spans = find_spans(np.asarray(alert_mask, dtype=bool), nT)
                error_spans = find_spans(np.asarray(error_mask, dtype=bool), nT)

                window_alerts += len(alert_spans)
                window_errors += len(error_spans)

                # very light pastel overlays so data stays visible
                overlay_spans(axp, x, alert_spans, color="#fff8b3", **overlay_kwargs)  # pale yellow
                overlay_spans(axp, x, error_spans,  color="#ffd6d6", **overlay_kwargs)  # pale red

            # ---- Series rendering ----
            if STACKED:
                # values per axis
                series = [
                    pd.to_numeric(test[ax], errors="coerce").fillna(0.0).to_numpy(dtype=float)[win_idx]
                    for ax in axes
                ]
                axis_colors = axis_palette[: len(series)]

                # soft stacked fill underneath (omit labels to avoid duplicates)
                axp.stackplot(x, *series, colors=axis_colors, alpha=0.60, zorder=1)

                # thin colored outlines with labels for legend
                for y, c, name in zip(series, axis_colors, axes):
                    axp.plot(x, y, color=c, linewidth=1.1, alpha=0.95, zorder=3, label=name)
            else:
                # non-stacked line mode with same palette
                axis_colors = axis_palette[: len(axes)]
                for c, name in zip(axis_colors, axes):
                    y = pd.to_numeric(test[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)[win_idx]
                    axp.plot(x, y, color=c, linewidth=1.2, alpha=0.95, zorder=3, label=name)

            axp.set_xlim(0, max(1, len(x) - 1))
            axp.set_xlabel(f"Step ({TICK_SECONDS}s per row)")
            axp.set_ylabel("Current (A)")
            axp.set_title(
                f"Live Playback — {WINDOW_ROWS*TICK_SECONDS}s Window\n"
                f"(ALERT=orange, ERROR=red; thresholds from training residuals)"
            )

            # readability tweaks
            axp.grid(True, which="both", linestyle="--", alpha=0.25)
            axp.margins(x=0, y=0.05)

            # Legend: axes (colored) + overlay patches (outside, compact)
            handles, labels = axp.get_legend_handles_labels()
            overlay_handles = [
                mpatches.Patch(color="#fff8b3", alpha=overlay_kwargs["alpha"], label="ALERT window"),
                mpatches.Patch(color="#ffd6d6", alpha=overlay_kwargs["alpha"], label="ERROR window"),
            ]
            handles += overlay_handles
            labels += ["ALERT window", "ERROR window"]

            axp.legend(
                handles=handles,
                labels=labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
                fontsize=8,
                title="Legend",
                title_fontsize=9,
            )

            # Status bar text
            axp.text(
                0.01,
                0.98,
                f"Window events → ALERT: {window_alerts} | ERROR: {window_errors}",
                transform=axp.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85),
            )

            fig.canvas.draw_idle()
            plt.pause(0.001)         # non-blocking UI refresh
            time.sleep(TICK_SECONDS) # pacing to mimic streaming
            i += 1

        print("✅ Playback finished.")
    except KeyboardInterrupt:
        print("⏹ Playback interrupted by user.")
    finally:
        plt.ioff()


if __name__ == "__main__":
    main()
