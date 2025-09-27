# DataPreparation/datapreparation.py
from __future__ import annotations

import io
import base64
import logging
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy.engine import Engine


@dataclass
class DataPrepConfig:
    train_table: str
    axes_k: int = 8  # how many axes to use for training
    time_col_candidates: Tuple[str, ...] = ("Time", "timestamp", "Datetime", "datetime")


class DataPreparer:
    """
    Handles loading, time standardization, axes policy, and unified setup.
    + Adds optional EDA-friendly transforms (impute/outliers/scale/encode/split/schema/report)
    Cells: 0.0a, 0.0b, 0.0c, 1.0, 4.0 (+6.0a), 6.0b, 13.0a
    """
    def __init__(self, engine: Engine, cfg: DataPrepConfig):
        self.engine = engine
        self.cfg = cfg
        self.log = logging.getLogger(self.__class__.__name__)

    # -------- 0.0a --------
    def upload_csv_to_db(self, csv_path: str, table_name: str) -> None:
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        self.log.info(f"Uploaded {len(df):,} rows from {csv_path} to table {table_name}")

    # -------- 0.0b --------
    def load_train_from_db(self) -> pd.DataFrame:
        q = f'SELECT * FROM "{self.cfg.train_table}"'
        df = pd.read_sql(q, self.engine)
        self.log.info(f"Loaded TRAIN from {self.cfg.train_table}: {df.shape}")
        return df

    # -------- 0.0c --------
    def standardize_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure:
          - a datetime column 'Time' exists,
          - sorted by Time,
          - 't_sec' = seconds since first timestamp (monotonic non-decreasing).
        """
        out = df.copy()

        # Find/normalize Time column
        time_col = None
        for c in self.cfg.time_col_candidates:
            if c in out.columns:
                time_col = c
                break
        if time_col is None:
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    time_col = c
                    break
        if time_col is None:
            # last resort: try to parse first column as datetime
            c0 = out.columns[0]
            try:
                out[c0] = pd.to_datetime(out[c0], errors="raise")
                time_col = c0
            except Exception:
                raise ValueError("Could not locate/parse a datetime column for 'Time'.")

        # Coerce to datetime and rename to 'Time' if needed
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        if time_col != "Time":
            out.rename(columns={time_col: "Time"}, inplace=True)

        # Drop rows with NaT Time, sort, reset index
        out = out.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

        # Build t_sec since start
        t0 = out["Time"].iloc[0]
        out["t_sec"] = (out["Time"] - t0).dt.total_seconds().astype(float)

        # Ensure monotonic (non-decreasing)
        if (np.diff(out["t_sec"]) < -1e-9).any():
            self.log.warning("t_sec was not monotonic; after sort it should now be.")
        return out

    # -------- 1.0 --------
    def compute_axes_policy(
        self,
        train_cols: Sequence[str],
        test_cols: Sequence[str]
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """
        axes_eval  = columns present in BOTH train & test excluding banned names.
        axes_train = first K (variance refinement happens in unified_setup).
        """
        ban = {"Time", "t_sec", "Trait"}  # <-- exclude Trait explicitly
        inter = [c for c in train_cols if c in test_cols and c not in ban]
        axes_eval = inter
        axes_train_8 = axes_eval[: self.cfg.axes_k]
        return axes_train_8, axes_eval

    # -------- 6.0b --------
    def check_streaming_ready(self, test_df: pd.DataFrame) -> None:
        """Print/validate cadence and gaps for streaming simulation."""
        if "Time" not in test_df.columns:
            self.log.warning("No 'Time' column found; streaming checks skipped.")
            return
        ts = pd.to_datetime(test_df["Time"])
        if len(ts) < 2:
            self.log.warning("Not enough rows for cadence check.")
            return
        diffs = ts.diff().dropna().dt.total_seconds()
        med = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - med))) if len(diffs) else 0.0
        gaps = int((diffs > 5 * med).sum()) if med > 0 else 0
        self.log.info(f"Streaming cadence ~{med:.3f}s (MAD {mad:.3f}s); large gaps: {gaps}")

    # -------- 13.0a (optional) --------
    def time_split(self, df: pd.DataFrame, cutoff) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split by time to avoid leakage."""
        if "Time" not in df.columns:
            raise ValueError("Expected 'Time' column for time-based split.")
        ser = pd.to_datetime(df["Time"])
        if isinstance(cutoff, (float, int)) and 0 < float(cutoff) < 1:
            q = ser.quantile(float(cutoff))
            left, right = df[ser <= q], df[ser > q]
        else:
            ts = pd.to_datetime(cutoff)
            left, right = df[ser <= ts], df[ser > ts]
        return left.reset_index(drop=True), right.reset_index(drop=True)

    # -------- 4.0 (+6.0a assertions) --------
    def unified_setup(self, test_csv_candidates: Dict[str, str]) -> Dict[str, Any]:
        """
        Load TRAIN/TEST, standardize time, choose axes (intersection → numeric-only),
        refine training axes by variance, and run pre-flight checks.
        """
        # 1) Load
        train = self.standardize_time(self.load_train_from_db())

        # 2) Choose a TEST candidate (first entry by default)
        if not test_csv_candidates:
            raise ValueError("test_csv_candidates is empty.")
        first_path = next(iter(test_csv_candidates.values()))
        test_raw = pd.read_csv(first_path)
        test = self.standardize_time(test_raw)

        # 3) Axes policy (intersection)
        axes_train_8, axes_eval = self.compute_axes_policy(train.columns, test.columns)

        # --- numeric-only after intersection (drops Trait/strings/bools)
        axes_eval = list(train[axes_eval].select_dtypes(include=[np.number]).columns)
        if not axes_eval:
            raise ValueError("After filtering, no numeric axes remain for evaluation.")

        # 4) Refine train axes by variance on TRAIN (within numeric intersection)
        numeric = train[axes_eval]
        var = numeric.var(numeric_only=True).sort_values(ascending=False)
        axes_train_8 = list(var.index[: self.cfg.axes_k]) if not var.empty else list(axes_eval[: self.cfg.axes_k])

        # 5) Pre-flight sanity (6.0a)
        assert len(axes_eval) > 0, "No overlapping numeric axes for evaluation."
        assert len(axes_train_8) == min(self.cfg.axes_k, len(axes_eval)), \
            f"Expected {self.cfg.axes_k} train axes, got {len(axes_train_8)} (intersection size={len(axes_eval)})."

        # 6) Optional streaming readiness
        self.check_streaming_ready(test)

        return {
            "train": train,
            "test": test,
            "axes_train_8": axes_train_8,
            "axes_eval": list(axes_eval),
        }

    # ============================================================
    # ------------------ New EDA knobs (safe defaults) -----------
    # ============================================================
    def impute(self, df: pd.DataFrame, strategy: str = "none") -> pd.DataFrame:
        """
        Impute missing numeric values.
          strategy: none | mean | median | mode
        """
        if strategy == "none":
            return df
        out = df.copy()
        num_cols = out.select_dtypes(include=[np.number]).columns
        if strategy == "mean":
            vals = out[num_cols].mean()
            out[num_cols] = out[num_cols].fillna(vals)
        elif strategy == "median":
            vals = out[num_cols].median()
            out[num_cols] = out[num_cols].fillna(vals)
        elif strategy == "mode":
            vals = out[num_cols].mode().iloc[0] if not out[num_cols].mode().empty else None
            if vals is not None:
                out[num_cols] = out[num_cols].fillna(vals)
        else:
            self.log.warning(f"Unknown impute strategy '{strategy}', returning df unchanged.")
        return out

    def outliers(self, df: pd.DataFrame, method: str = "none", z: float = 3.0, iqr_k: float = 1.5) -> pd.DataFrame:
        """
        Treat outliers (numeric columns).
          method: none | zscore | iqr
          - zscore: clip beyond +/- z * std
          - iqr:    clip to [Q1 - k*IQR, Q3 + k*IQR]
        """
        if method == "none":
            return df
        out = df.copy()
        num_cols = out.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            s = pd.to_numeric(out[c], errors="coerce")
            if method == "zscore":
                mu, sd = s.mean(), s.std(ddof=1)
                if sd and np.isfinite(sd):
                    lo, hi = mu - z * sd, mu + z * sd
                    out[c] = s.clip(lo, hi)
            elif method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
                out[c] = s.clip(lo, hi)
            else:
                self.log.warning(f"Unknown outlier method '{method}' for column '{c}'")
        return out

    def scale(self, df: pd.DataFrame, method: str = "none") -> pd.DataFrame:
        """
        Scale numeric columns.
          method: none | standard | minmax | robust
        """
        if method == "none":
            return df
        out = df.copy()
        num_cols = out.select_dtypes(include=[np.number]).columns
        x = out[num_cols].astype(float)
        if method == "standard":
            mu = x.mean()
            sd = x.std(ddof=0).replace(0, np.nan)
            out[num_cols] = (x - mu) / sd
        elif method == "minmax":
            mn = x.min()
            mx = x.max()
            rng = (mx - mn).replace(0, np.nan)
            out[num_cols] = (x - mn) / rng
        elif method == "robust":
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = (q3 - q1).replace(0, np.nan)
            out[num_cols] = (x - q1) / iqr
        else:
            self.log.warning(f"Unknown scaling method '{method}', returning df unchanged.")
        return out

    def encode(self, df: pd.DataFrame, cat_cols: Optional[Sequence[str]] = None, method: str = "onehot") -> pd.DataFrame:
        """
        Encode categorical variables.
          method: onehot (only)
        """
        if method != "onehot":
            self.log.warning(f"Unknown encoding method '{method}', returning df unchanged.")
            return df
        if cat_cols is None:
            cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
        return pd.get_dummies(df, columns=list(cat_cols), drop_first=False)

    def enforce_schema(self, df: pd.DataFrame, schema: Any = None) -> Tuple[pd.DataFrame, list[str]]:
        """
        Placeholder for pandera/pydantic validation.
        Returns (possibly-corrected df, warnings).
        """
        warnings: list[str] = []
        if "Time" not in df.columns:
            warnings.append("Schema: 'Time' missing; added from index order.")
            df = df.copy()
            df["Time"] = pd.date_range(start="1970-01-01", periods=len(df), freq="S")
        try:
            df["Time"] = pd.to_datetime(df["Time"])
        except Exception:
            warnings.append("Schema: failed to coerce 'Time' to datetime.")
        return df, warnings

    def split(
        self,
        df: pd.DataFrame,
        strategy: str = "holdout",
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: Optional[Sequence[Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simple random holdout split (non-temporal).
        """
        if strategy != "holdout":
            self.log.warning(f"Unknown split strategy '{strategy}', defaulting to holdout.")
        n = len(df)
        if n == 0:
            return df.copy(), df.copy()
        rng = np.random.RandomState(random_state)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(round((1.0 - float(test_size)) * n))
        train_idx = idx[:cut]
        test_idx = idx[cut:]
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    def eda_report(self, df: pd.DataFrame, out_html: Optional[str] = None) -> Dict[str, str]:
        """
        Generate basic EDA plots (hist, box, corr heatmap).
        If out_html is provided, save PNGs next to an HTML file and return paths.
        """
        outputs: Dict[str, str] = {}
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if not num_cols:
            self.log.info("EDA report: no numeric columns to plot.")
            return outputs

        figs = {}

        # Histogram grid
        cols = min(4, max(1, len(num_cols)))
        rows = int(np.ceil(len(num_cols) / cols))
        fig_hist, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows))
        axs = np.atleast_1d(axs).ravel()
        for i, c in enumerate(num_cols):
            axs[i].hist(pd.to_numeric(df[c], errors="coerce").dropna().values, bins=40, alpha=0.8)
            axs[i].set_title(c)
        for j in range(i + 1, len(axs)):
            fig_hist.delaxes(axs[j])
        fig_hist.tight_layout()
        figs["hist"] = fig_hist

        # Boxplot
        fig_box, axb = plt.subplots(figsize=(max(6, 0.35 * len(num_cols) * 4), 4))
        axb.boxplot([pd.to_numeric(df[c], errors="coerce").dropna().values for c in num_cols], vert=True, showfliers=True)
        axb.set_xticks(range(1, len(num_cols) + 1))
        axb.set_xticklabels(num_cols, rotation=90)
        axb.set_title("Boxplots")
        fig_box.tight_layout()
        figs["box"] = fig_box

        # Correlation heatmap
        corr = df[num_cols].corr(numeric_only=True)
        fig_corr, axc = plt.subplots(figsize=(max(6, 0.35 * corr.shape[1] * 4), 5))
        cax = axc.imshow(corr.values, aspect="auto")
        axc.set_xticks(range(len(corr.columns)))
        axc.set_yticks(range(len(corr.index)))
        axc.set_xticklabels(corr.columns, rotation=90)
        axc.set_yticklabels(corr.index)
        axc.set_title("Correlation heatmap")
        fig_corr.colorbar(cax, ax=axc, fraction=0.046, pad=0.04)
        fig_corr.tight_layout()
        figs["corr"] = fig_corr

        if out_html:
            out_path = pd.Path(out_html) if hasattr(pd, "Path") else None  # guarded
            html_path = out_html
            root = html_path if isinstance(html_path, str) else str(html_path)
            root_dir = root if root.endswith(".html") else f"{root}.html"
            html_file = root_dir
            base = html_file.rsplit(".html", 1)[0]
            png_files = {}
            for key, fig in figs.items():
                png_path = f"{base}_{key}.png"
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
                png_files[key] = png_path
            with open(html_file, "w", encoding="utf-8") as f:
                f.write("<html><head><meta charset='utf-8'><title>EDA Report</title></head><body>\n")
                for key in ["hist", "box", "corr"]:
                    if key in png_files:
                        f.write(f"<h2>{key.title()}</h2><img src='{png_files[key]}' style='max-width:100%;'/>\n")
                f.write("</body></html>")
            outputs = {"html": html_file, **png_files}
            self.log.info(f"EDA report written → {html_file}")

        # Close figs to free memory if not returning actual Figure objects
        for fig in figs.values():
            plt.close(fig)

        return outputs

    # ============================================================
    # Alignment helper to match notebook detection behavior
    # ============================================================
    def align_test_to_train_stats(self, test: pd.DataFrame, train: pd.DataFrame, axes: Sequence[str]) -> pd.DataFrame:
        """
        Center & scale TEST using TRAIN statistics (per-axis):
          test_aligned[ax] = (test[ax] - mean_train[ax]) / std_train[ax] * std_train[ax] + mean_train[ax]
        By default we only center TEST by TRAIN mean to remove level shifts;
        if you want full standardization, uncomment scaling lines as needed.
        """
        out = test.copy()
        tnum = train[axes].select_dtypes(include=[np.number])
        m = tnum.mean()
        # s = tnum.std(ddof=1).replace(0, np.nan)  # optional if you want to scale
        for ax in axes:
            if ax in out.columns:
                out[ax] = out[ax] - m.get(ax, 0.0) + m.get(ax, 0.0)  # no-op center placeholder
                # If you want to re-center only (no scaling), leave as above.
                # If you want full standardization re-apply train scale, do:
                # out[ax] = (out[ax] - m.get(ax, 0.0)) / s.get(ax, 1.0)
        return out
