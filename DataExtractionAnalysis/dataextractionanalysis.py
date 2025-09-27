# DataExtractionAnalysis/dataextractionanalysis.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from dotenv import load_dotenv
from sqlalchemy.engine import Engine


# ============================
# Ingestion / source utilities
# ============================
@dataclass
class SourceConfig:
    env: str            # dev | qa | prod
    kind: str           # database | api | fs
    dsn: str | None = None       # database URL / DSN
    root: str | None = None      # filesystem root
    base_url: str | None = None  # API base URL


class DataExplorer:
    """
    EDA + distribution overlays + light ingestion scaffolding.
    Cells: 1.1 (axes_healthcheck), 2.0b/2.0c/3.0a/3.0b (overlay_train_vs_test)
    """

    def __init__(self, engine: Optional[Engine] = None):
        self.engine = engine
        self.log = logging.getLogger(self.__class__.__name__)

    # -------- Source detection (env-aware) --------
    def detect_source(self, env: Optional[str] = None) -> SourceConfig:
        """
        Read source config from environment or .env:
          ENV=dev|qa|prod
          SOURCE_KIND=database|api|fs
          DATABASE_URL=...
          FS_ROOT=/path/to/data
          API_BASE_URL=https://...
        """
        load_dotenv(override=True)
        env_val = (env or os.getenv("ENV", "dev")).lower()
        kind = os.getenv("SOURCE_KIND", "database").lower()
        cfg = SourceConfig(
            env=env_val,
            kind=kind,
            dsn=os.getenv("DATABASE_URL"),
            root=os.getenv("FS_ROOT"),
            base_url=os.getenv("API_BASE_URL"),
        )
        self.log.info(f"Detected source: env={cfg.env} kind={cfg.kind}")
        return cfg

    # -------- Schema validation (stub) --------
    def validate_schema(self, df: pd.DataFrame, schema: Any = None) -> tuple[pd.DataFrame, list[str]]:
        """
        Placeholder for pandera/pydantic schema checks.
        Returns (possibly-corrected df, warnings).
        """
        warnings: list[str] = []
        if df is None or df.empty:
            warnings.append("Input DataFrame is empty.")
            return df, warnings

        # Example light checks
        if "Time" not in df.columns:
            warnings.append("Column 'Time' not found; attempting to coerce from index.")
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "Time"})
            else:
                # create synthetic time if completely missing
                df = df.copy()
                df["Time"] = pd.date_range(start="1970-01-01", periods=len(df), freq="S")

        # Ensure datetime type
        try:
            df["Time"] = pd.to_datetime(df["Time"])
        except Exception:
            warnings.append("Failed to parse 'Time' column to datetime.")

        return df, warnings

    # -------- Snapshotting / versioning --------
    def snapshot(self, df: pd.DataFrame, out_dir: str | Path, tag: str = "raw") -> str:
        """
        Save a raw CSV snapshot with timestamp for reproducibility.
        Returns file path as string.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"{tag}-{ts}.csv"
        df.to_csv(path, index=False)
        self.log.info(f"Snapshot saved → {path}")
        return str(path)

    # -------- Incremental / streaming support (no-op for now) --------
    def incrementals(self, df: pd.DataFrame, last_watermark: Any | None = None) -> pd.DataFrame:
        """
        Placeholder: return full df. In production, filter rows newer than 'last_watermark'.
        """
        return df

    # -------- PII masking --------
    def redact_pii(self, df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        """
        Mask sensitive fields. This is a simple stub; swap for tokenization/vault lookups as needed.
        """
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = out[c].astype(str).str.replace(r".", "•", regex=True)
        return out

    # -------- Ingestion monitoring hook --------
    def ingest_monitor(self, status: str, meta: Dict[str, Any] | None = None) -> None:
        """
        Log ingestion status; later, integrate with Prometheus/MLflow.
        """
        meta = meta or {}
        self.log.info(f"[INGEST] status={status} meta={meta}")

    # -------- Initial profiling --------
    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Basic profile: row count, dtypes, nulls per column.
        Logs and returns the profile dictionary.
        """
        prof: Dict[str, Any] = {
            "rows": int(len(df)),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "nulls": {c: int(df[c].isna().sum()) for c in df.columns},
        }
        self.log.info(f"Profile → rows={prof['rows']}")
        return prof

    # ============================================================
    # Existing EDA helpers (kept intact)
    # ============================================================
    # ---------- 1.1 ----------
    def axes_healthcheck(self, df: pd.DataFrame, axes: Sequence[str]) -> pd.DataFrame:
        """
        Quick per-axis data quality table.
        Columns:
          - dtype, count, n_null, null_pct
          - n_unique
          - min, q1, median, mean, q3, max, std
        """
        rows: list[Dict[str, Any]] = []
        for ax in axes:
            # SAFER access: only convert if column exists
            if ax in df.columns:
                s = pd.to_numeric(df[ax], errors="coerce")
            else:
                s = None

            if s is None:
                rows.append({
                    "axis": ax, "dtype": "missing", "count": 0, "n_null": np.nan, "null_pct": np.nan,
                    "n_unique": np.nan, "min": np.nan, "q1": np.nan, "median": np.nan,
                    "mean": np.nan, "q3": np.nan, "max": np.nan, "std": np.nan
                })
                continue

            cnt = int(s.shape[0])
            n_null = int(s.isna().sum())
            null_pct = (n_null / cnt * 100.0) if cnt else np.nan
            nz = s.dropna()
            rows.append({
                "axis": ax,
                "dtype": str(df[ax].dtype) if ax in df.columns else "missing",
                "count": cnt,
                "n_null": n_null,
                "null_pct": round(null_pct, 3) if np.isfinite(null_pct) else np.nan,
                "n_unique": int(nz.nunique()) if not nz.empty else 0,
                "min": float(nz.min()) if not nz.empty else np.nan,
                "q1": float(nz.quantile(0.25)) if not nz.empty else np.nan,
                "median": float(nz.median()) if not nz.empty else np.nan,
                "mean": float(nz.mean()) if not nz.empty else np.nan,
                "q3": float(nz.quantile(0.75)) if not nz.empty else np.nan,
                "max": float(nz.max()) if not nz.empty else np.nan,
                "std": float(nz.std(ddof=1)) if len(nz) > 1 else np.nan,
            })
        rep = pd.DataFrame(rows)
        order = ["axis","dtype","count","n_null","null_pct","n_unique","min","q1","median","mean","q3","max","std"]
        return rep[order]

    # ---------- 2.0b/2.0c/3.0a/3.0b ----------
    def overlay_train_vs_test(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        axes: Sequence[str],
        *,
        bins: int = 40,
        sharex: bool = False,
        sharey: bool = False,
        density: bool = True,
        grid: bool = True,
    ) -> Figure:
        """
        Histogram/KDE overlay per axis comparing TRAIN vs TEST.
        Returns a matplotlib Figure.
        """
        k = max(1, len(axes))
        ncols = 3 if k >= 3 else k
        nrows = int(np.ceil(k / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), sharex=sharex, sharey=sharey)
        axs = np.atleast_1d(axs).ravel()

        for i, ax_name in enumerate(axes):
            axp = axs[i]
            tr = pd.to_numeric(train.get(ax_name), errors="coerce").dropna()
            te = pd.to_numeric(test.get(ax_name), errors="coerce").dropna()

            # Choose common range for fair comparison
            lo = np.nanmin([tr.min() if len(tr) else np.nan, te.min() if len(te) else np.nan])
            hi = np.nanmax([tr.max() if len(tr) else np.nan, te.max() if len(te) else np.nan])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                common_range = None
            else:
                pad = 0.02 * (hi - lo if hi > lo else 1.0)
                common_range = (lo - pad, hi + pad)

            if len(tr):
                axp.hist(tr, bins=bins, alpha=0.4, density=density, range=common_range, label="TRAIN")
                try:
                    tr.plot(kind="kde", ax=axp, linewidth=1.2)
                except Exception:
                    pass

            if len(te):
                axp.hist(te, bins=bins, alpha=0.4, density=density, range=common_range, label="TEST")
                try:
                    te.plot(kind="kde", ax=axp, linewidth=1.2, linestyle="--")
                except Exception:
                    pass

            axp.set_title(ax_name)
            if grid:
                axp.grid(True, alpha=0.25)
            if i == 0:
                axp.legend()

        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle("TRAIN vs TEST distributions", y=0.995)
        fig.tight_layout()
        return fig

    # ============================
    # Optional loaders used by orchestrator (kept simple)
    # ============================
    def load_train_from_db(self, table: str) -> pd.DataFrame:
        if self.engine is None:
            raise RuntimeError("SQLAlchemy engine is not set on DataExplorer.")
        df = pd.read_sql_table(table, con=self.engine)
        df, _ = self.validate_schema(df)
        return df

    def load_test_from_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df, _ = self.validate_schema(df)
        return df
