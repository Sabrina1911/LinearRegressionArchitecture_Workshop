# TrainedMLModel/trainedmlmodel.py
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import pandas as pd


class ArtifactStore:
    """
    Save/load thresholds, models, and event logs.

    Notes:
    - Thresholds are saved WITH index (axis as index) and loaded with index_col=0.
      If a legacy CSV (without index) is found, we fallback to 'axis' column if present.
    - Simple model registry stubs:
        * save_model / load_model for single models + JSON sidecar metadata
        * save_models / load_models for a dict of per-axis models (bundle)
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts" / "models").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------
    def save_thresholds(self, thresholds: pd.DataFrame, name: str = "thresholds.csv") -> Path:
        """
        Save thresholds WITH index (axis names).
        """
        out_path = self.root / "artifacts" / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        thresholds.to_csv(out_path, index=True)  # keep axis index
        return out_path

    def load_thresholds(self, name: str = "thresholds.csv") -> pd.DataFrame:
        """
        Load thresholds and ensure axis index is set.
        Supports legacy files saved without index (expects an 'axis' column).
        """
        in_path = self.root / "artifacts" / name
        df = pd.read_csv(in_path, index_col=0)  # preferred path (index saved)

        # Fallback: if an 'axis' column exists, make it the index
        if "axis" in df.columns:
            # Only reset index to 'axis' if current index doesn't already look like axes
            if df.index.dtype.kind in ("i", "u", "f") or df.index.name not in ("axis", "Axis"):
                df = df.set_index("axis")

        # Normalize T column name
        if "T_sec" not in df.columns and "T" in df.columns:
            df = df.rename(columns={"T": "T_sec"})

        # Ensure consistent dtypes (optional but nice)
        for c in ("MinC", "MaxC", "T_sec"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Stringify index to avoid later mismatches like 1 vs "1"
        df.index = df.index.map(str)

        return df

    def finalize_thresholds(self, thresholds: pd.DataFrame, **provenance: Any) -> pd.DataFrame:
        """
        Attach provenance columns (e.g., cadence_s=1.0, strategy='IQR', created_by='orchestrator').
        Scalar values are broadcast; equal-length sequences are used as-is.
        """
        out = thresholds.copy()
        n = len(out)
        for k, v in provenance.items():
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                out[k] = v if len(v) == n else [v] * n
            else:
                out[k] = [v] * n
        return out

    # ------------------------------------------------------------------
    # Model registry (simple stubs)
    # ------------------------------------------------------------------
    def _ts(self) -> str:
        return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    def save_model(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        """
        Persist a single model under artifacts/models with a versioned filename
        and a JSON sidecar containing metadata (params/metrics/etc).

        Returns: (model_path, meta_path)
        """
        models_dir = self.root / "artifacts" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        version = self._ts()
        base = f"{name}-{version}"
        model_path = models_dir / f"{base}.pkl"
        meta_path = models_dir / f"{base}.json"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        meta = {
            "name": name,
            "version": version,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
        }
        if metadata:
            meta.update(metadata)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return model_path, meta_path

    def load_model(self, name_or_path: str) -> Any:
        """
        Load a single model. If `name_or_path` is a path to a .pkl file, load it.
        If it's a registry name, load the latest version by timestamp.
        """
        p = Path(name_or_path)
        models_dir = self.root / "artifacts" / "models"

        if p.suffix == ".pkl" and p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)

        # Treat as a registry name; find the latest version
        candidates = sorted(models_dir.glob(f"{name_or_path}-*.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No model found for name '{name_or_path}' in {models_dir}")
        latest = candidates[-1]
        with open(latest, "rb") as f:
            return pickle.load(f)

    def save_models(self, models: Dict[str, Any], bundle_name: str = "models.pkl", metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save a dict of per-axis models as a single bundle (pickle) and a JSON sidecar.
        """
        models_dir = self.root / "artifacts" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = models_dir / bundle_name
        with open(bundle_path, "wb") as f:
            pickle.dump(models, f)

        meta = {
            "bundle": bundle_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "num_models": len(models),
            "keys": list(models.keys()),
        }
        if metadata:
            meta.update(metadata)

        with open(models_dir / (Path(bundle_name).stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return bundle_path

    def load_models(self, bundle_name: str = "models.pkl") -> Dict[str, Any]:
        """
        Load a dict of per-axis models from a bundle.
        """
        bundle_path = self.root / "artifacts" / "models" / bundle_name
        if not bundle_path.exists():
            # Fallback: try to find any .pkl in the models dir (last modified)
            candidates = sorted((self.root / "artifacts" / "models").glob("*.pkl"))
            if not candidates:
                raise FileNotFoundError(f"No model bundle found at {bundle_path}")
            bundle_path = candidates[-1]
        with open(bundle_path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Event logs
    # ------------------------------------------------------------------
    def clear_events_log(self, name: str = "events.csv") -> bool:
        """
        Delete the events logfile if it exists. Returns True if deleted.
        """
        p = self.root / "logs" / name
        if p.exists():
            p.unlink()
            return True
        return False

    def append_events(self, events_df: pd.DataFrame, name: str = "events.csv") -> Path:
        """
        Append events to logs/<name>. Creates the file with a header if it doesn't exist.
        Returns the path written.
        Expected columns: ['axis','label','start_time','end_time'] (extras are preserved).
        """
        p = self.root / "logs" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        header = not p.exists()
        events_df.to_csv(p, mode="a", index=False, header=header)
        return p

    def read_events(self, name: str = "events.csv") -> pd.DataFrame:
        """
        Read the events log if present; else return empty DataFrame with standard columns.
        """
        p = self.root / "logs" / name
        if p.exists():
            return pd.read_csv(p, parse_dates=["start_time", "end_time"], infer_datetime_format=True)
        return pd.DataFrame(columns=["axis", "label", "start_time", "end_time"])
