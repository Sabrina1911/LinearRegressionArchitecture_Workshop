# ModelEvaluationValidation/modelevaluationvalidation.py
from __future__ import annotations

import math
from typing import Sequence, Dict, Any, Tuple, List, Callable  # <-- added Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class ModelEvaluator:
    """
    Residuals, thresholds, symmetric event detection (with dwell), overlays, and summaries.

    Assumptions
    ----------
    - DataFrames have columns: "Time" (datetime-like) and "t_sec" (float seconds).
    - `models` is a dict: {axis_name: fitted_model}, each model supports .predict(X) with X[['t_sec']].
    - `thresholds` is a DataFrame with index = axis names and columns: ['MinC','MaxC','T_sec'].
    - `events` DataFrame uses columns: ['axis','label','start_time','end_time'].
    """

    # ============================================================
    # Small helpers (kept public-ish for orchestrator compatibility)
    # ============================================================
    def _coerce_time(self, df: pd.DataFrame) -> pd.Series:
        """Return a pandas Series of timestamps from df['Time']."""
        assert "Time" in df.columns, "Expected 'Time' column"
        return pd.to_datetime(df["Time"])

    def _make_time_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure there's a 't_sec' column from Time=0. If already present, reuse.
        Returns a DataFrame with a single column 't_sec' for model predict().
        """
        if "t_sec" in df.columns:
            return df[["t_sec"]]
        t = self._coerce_time(df)
        t0 = t.iloc[0]
        tsec = (t - t0).dt.total_seconds().astype(float)
        return pd.DataFrame({"t_sec": tsec}, index=df.index)

    # ============================================================
    # 1) Residuals helper (parity with notebook: post-alignment)
    # ============================================================
    def compute_residuals(
        self,
        df: pd.DataFrame,
        models: Dict[str, Any],
        axes: Sequence[str],
    ) -> pd.DataFrame:
        """
        Return residuals DataFrame: same index as df, columns=axes, values = y - y_hat.
        """
        X = self._make_time_feature(df)  # robust to missing 't_sec'
        out = {}
        for ax in axes:
            if ax not in df.columns:
                # If column missing, retain NaNs for clarity
                out[ax] = np.full(len(df), np.nan)
                continue
            m = models.get(ax, None)
            if m is None:
                out[ax] = np.full(len(df), np.nan)
            else:
                yhat = np.asarray(m.predict(X)).reshape(-1)
                out[ax] = df[ax].to_numpy().reshape(-1) - yhat
        return pd.DataFrame(out, index=df.index)

    # ============================================================
    # 2) Build thresholds from TRAIN residuals (robust IQR)
    # ============================================================
    def build_thresholds_from_residuals(
        self,
        train_residuals: pd.DataFrame,
        cadence_s: float,
        options: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Strategy:
          - Robust spread via IQR (Q3 - Q1)
          - MinC = Q1 - k*IQR, MaxC = Q3 + k*IQR (default k=1.5)
          - T_sec = max(cadence_s * t_mult, cadence_s)  (default t_mult=3.0)
        """
        if train_residuals is None or train_residuals.empty:
            return pd.DataFrame(columns=["MinC", "MaxC", "T_sec"])

        options = options or {}
        k = float(options.get("iqr_k", 1.5))
        t_mult = float(options.get("t_mult", 3.0))

        rows: List[Dict[str, float]] = []
        for ax in train_residuals.columns:
            s = pd.to_numeric(train_residuals[ax], errors="coerce").dropna()
            if s.empty:
                rows.append({"axis": ax, "MinC": np.nan, "MaxC": np.nan, "T_sec": np.nan})
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = max(q3 - q1, 1e-9)
            minc = float(q1 - k * iqr)
            maxc = float(q3 + k * iqr)
            tsec = float(max(cadence_s * t_mult, cadence_s))
            rows.append({"axis": ax, "MinC": minc, "MaxC": maxc, "T_sec": tsec})

        return pd.DataFrame(rows).set_index("axis")

    # ============================================================
    # 3) Symmetric event detection with dwell (two-sided)
    # ============================================================
    def detect_events(
        self,
        test: pd.DataFrame,
        models: Dict[str, Any],
        thresholds: pd.DataFrame,
        axes: Sequence[str],
        *,
        cadence_s: float,
        on_event: Callable[[Dict[str, Any]], None] | None = None,  # <-- NEW optional callback
    ) -> pd.DataFrame:
        """
        Emit an event when residual stays beyond |MinC| or |MaxC| for >= T_sec.
        Two-sided logic:
            ERROR if (res >= MaxC) or (res <= -MaxC)
            ALERT if (res >= MinC) or (res <= -MinC)
        Durations are measured in *seconds* via the 'Time' column.

        If provided, `on_event(event_row_dict)` is called for each finalized span.
        """
        assert "Time" in test.columns, "Expected 'Time' column in test"
        res_df = self.compute_residuals(test, models, axes)
        t = self._coerce_time(test)

        def _label(v: float, minc: float, maxc: float) -> str | None:
            if np.isnan(v):
                return None
            if v >= maxc or v <= -maxc:
                return "ERROR"
            if v >= minc or v <= -minc:
                return "ALERT"
            return None

        events: List[Dict[str, Any]] = []

        for ax in axes:
            if ax not in res_df.columns or ax not in thresholds.index:
                continue

            # --- normalize thresholds to magnitudes (guard against signed CSVs) ---
            thr = thresholds.loc[ax]
            minc = abs(float(thr["MinC"]))
            maxc = abs(float(thr["MaxC"]))
            T = float(thr["T_sec"])

            start_idx = None
            current_label = None

            for i, v in enumerate(res_df[ax].to_numpy()):
                lab = _label(v, minc, maxc)

                if current_label is None:
                    if lab is not None:
                        start_idx = i
                        current_label = lab
                else:
                    # if label drops to None or changes class, check dwell then close
                    if (lab is None) or (lab != current_label):
                        if start_idx is not None:
                            dur_sec = (t.iloc[i - 1] - t.iloc[start_idx]).total_seconds()
                            if dur_sec >= T:
                                row = {
                                    "axis": ax,
                                    "label": current_label,
                                    "start_time": t.iloc[start_idx],
                                    "end_time": t.iloc[i - 1],
                                }
                                events.append(row)
                                if on_event:
                                    try:
                                        on_event(row)
                                    except Exception:
                                        pass
                        # start a new candidate if still in an abnormal label
                        start_idx = i if lab is not None else None
                        current_label = lab
                    # else: label unchanged → keep accumulating

            # tail flush
            if current_label is not None and start_idx is not None:
                dur_sec = (t.iloc[-1] - t.iloc[start_idx]).total_seconds()
                if dur_sec >= T:
                    row = {
                        "axis": ax,
                        "label": current_label,
                        "start_time": t.iloc[start_idx],
                        "end_time": t.iloc[-1],
                    }
                    events.append(row)
                    if on_event:
                        try:
                            on_event(row)
                        except Exception:
                            pass

        if not events:
            return pd.DataFrame(columns=["axis", "label", "start_time", "end_time"])
        return pd.DataFrame(events)

    # ============================================================
    # 4) Overlay grid (prediction vs actual) with shaded events
    # ============================================================
    def overlay_grid(
        self,
        df: pd.DataFrame,
        models: Dict[str, Any],
        events: pd.DataFrame,
        axes: Sequence[str],
        ncols: int = 2,
    ) -> Figure:
        """
        Per-axis plots of actual vs prediction; shades ALERT/ERROR spans from `events`.
        """
        assert "Time" in df.columns, "Need 'Time' column"
        X = self._make_time_feature(df)
        k = len(axes)
        ncols = max(1, ncols)
        nrows = max(1, math.ceil(k / ncols))
        fig = plt.figure(figsize=(5.6 * ncols, 3.2 * nrows), constrained_layout=True)

        for i, axname in enumerate(axes, start=1):
            a = fig.add_subplot(nrows, ncols, i)

            # actual
            if axname in df.columns:
                a.plot(df["Time"], df[axname], label="actual", linewidth=1.1)

            # prediction
            if axname in models and models[axname] is not None:
                yhat = np.asarray(models[axname].predict(X)).reshape(-1)
                a.plot(df["Time"], yhat, linestyle="--", label="pred", linewidth=1.0)

            # shade spans from events
            if isinstance(events, pd.DataFrame) and not events.empty:
                ev_ax = events[events["axis"].astype(str) == str(axname)]
                for _, r in ev_ax.iterrows():
                    # color by label (optional)
                    color = {"ALERT": None, "ERROR": None}.get(str(r.get("label")), None)
                    alpha = 0.15 if str(r.get("label")) == "ALERT" else 0.25
                    a.axvspan(r["start_time"], r["end_time"], alpha=alpha, color=color)

            a.set_title(axname)
            a.grid(True, alpha=0.25)
            if i == 1:
                a.legend(fontsize=8)

        return fig

    # ============================================================
    # 5) Optional: TRAIN vs TEST distribution overlay (EDA)
    # ============================================================
    def overlay_distributions(
        self, train: pd.DataFrame, test: pd.DataFrame, axes: Sequence[str]
    ) -> Figure:
        """
        Simple per-axis hist overlay (TRAIN vs TEST) as a quick EDA view.
        """
        cols = min(4, max(1, len(axes)))
        rows = max(1, math.ceil(len(axes) / cols))
        fig = plt.figure(figsize=(4.8 * cols, 3.2 * rows), constrained_layout=True)

        for i, axname in enumerate(axes, start=1):
            a = fig.add_subplot(rows, cols, i)
            if axname in train.columns:
                a.hist(
                    pd.to_numeric(train[axname], errors="coerce").dropna().values,
                    bins=40,
                    alpha=0.5,
                    label="TRAIN",
                    density=True,
                )
            if axname in test.columns:
                a.hist(
                    pd.to_numeric(test[axname], errors="coerce").dropna().values,
                    bins=40,
                    alpha=0.5,
                    label="TEST",
                    density=True,
                )
            a.set_title(axname)
            a.legend(fontsize=8)
            a.grid(True, alpha=0.2)

        return fig

    # ============================================================
    # 6) Decluttered multi-axis summary (with event spans)
    # ============================================================
    def summary_dashboard(
        self,
        df: pd.DataFrame,
        events: pd.DataFrame,
        axes: Sequence[str],
        *,
        show_topk: int = 4,
        smooth_win: int = 5,
        decimate: int = 3,
        merge_gap_sec: float = 6.0,
        robust_ylim: bool = True,
    ) -> Figure:
        """
        Decluttered multi-axis summary:
          - Pick top-k axes by std (or all if k >= len(axes))
          - Optional rolling-median smoothing
          - Decimation (every Nth point)
          - Merge close spans (gap <= merge_gap_sec)
          - Robust y-limits (1–99th percentiles) to avoid outlier blow-ups
        """
        assert "Time" in df.columns, "Expected 'Time' column."
        axes_use = list(axes)

        # top-k by std
        if show_topk and show_topk < len(axes_use):
            stds = df[axes_use].std(numeric_only=True).sort_values(ascending=False)
            axes_use = list(stds.index[:show_topk])

        plot_df = df.copy()
        # smoothing
        if smooth_win and smooth_win > 1:
            plot_df[axes_use] = plot_df[axes_use].rolling(smooth_win, center=True, min_periods=1).median()
        # decimation
        if decimate and decimate > 1:
            plot_df = plot_df.iloc[::decimate, :]

        # merge spans (optional)
        merged_events = events.copy()
        if isinstance(merged_events, pd.DataFrame) and not merged_events.empty and merge_gap_sec and merge_gap_sec > 0:
            merged = []
            for ax in merged_events["axis"].unique():
                sub = merged_events[merged_events["axis"] == ax].sort_values("start_time")
                last = None
                for _, r in sub.iterrows():
                    if (last is not None) and ((r["start_time"] - last["end_time"]).total_seconds() <= merge_gap_sec) and (r["label"] == last["label"]):
                        # extend
                        last["end_time"] = max(last["end_time"], r["end_time"])
                    else:
                        if last is not None:
                            merged.append(last)
                        last = r.to_dict()
                if last is not None:
                    merged.append(last)
            merged_events = pd.DataFrame(merged) if merged else merged_events

        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
        ax_main = fig.add_subplot(1, 1, 1)

        for ax_name in axes_use:
            if ax_name in plot_df.columns:
                ax_main.plot(plot_df["Time"], plot_df[ax_name], label=ax_name, linewidth=1.0)

        # draw event spans
        if isinstance(merged_events, pd.DataFrame) and not merged_events.empty:
            for _, r in merged_events.iterrows():
                alpha = 0.12 if str(r.get("label")) == "ALERT" else 0.2
                ax_main.axvspan(r["start_time"], r["end_time"], alpha=alpha)

        # robust y-lims
        if robust_ylim:
            vals = plot_df[axes_use].to_numpy().ravel()
            vals = vals[~np.isnan(vals)]
            if vals.size > 0:
                lo, hi = np.percentile(vals, [1, 99])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ax_main.set_ylim(lo, hi)

        ax_main.grid(True, alpha=0.25)
        ax_main.legend(ncol=min(4, len(axes_use)), fontsize=8)
        ax_main.set_title("Summary (decluttered)")
        return fig
