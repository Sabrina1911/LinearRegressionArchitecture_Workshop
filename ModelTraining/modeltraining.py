# ModelTraining/modeltraining.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Dict, Any, Optional

import numpy as np
import pandas as pd

# Light type import (no runtime dependency)
try:
    from ModelSelection.modelselection import ModelConfig  # for typing / config interop
except Exception:  # pragma: no cover
    @dataclass
    class ModelConfig:  # fallback stub
        family: str = "linear"
        params: Optional[dict] = None

# Try to use scikit-learn; if not available, fall back to a minimal linear model
try:
    from sklearn.linear_model import LinearRegression  # type: ignore
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False

    class LinearRegression:  # minimal drop-in for 1D X
        def __init__(self):
            self.coef_: Optional[np.ndarray] = None
            self.intercept_: float = 0.0

        def fit(self, X: np.ndarray, y: np.ndarray):
            # X: (n,1) of t_sec, y: (n,)
            X = np.asarray(X).reshape(-1, 1)
            y = np.asarray(y).reshape(-1)
            x = X[:, 0]
            # simple least squares using polyfit
            b1, b0 = np.polyfit(x, y, deg=1)  # y ≈ b1*x + b0
            self.coef_ = np.array([b1], dtype=float)
            self.intercept_ = float(b0)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            X = np.asarray(X).reshape(-1, 1)
            return self.intercept_ + X[:, 0] * (self.coef_[0] if self.coef_ is not None else 0.0)


class MeanRegressor:
    """Baseline model: predicts the training mean."""
    def __init__(self):
        self.mu_: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MeanRegressor":
        self.mu_ = float(np.nanmean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.mu_ is None:
            raise RuntimeError("Model not fitted")
        return np.full(shape=(len(X),), fill_value=self.mu_, dtype=float)


class ModelTrainer:
    """
    Train per-axis estimators according to a ModelConfig; add preds/residuals helpers.

    Assumptions
    -----------
    - `train_df` / `df` have columns: "t_sec" (float seconds) and each axis in `axes`.
    - Each model exposes .predict(X) with X as a DataFrame/array containing 't_sec'.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.log = logging.getLogger(self.__class__.__name__)

    # ---------------- internal factory ----------------
    def _make_estimator(self, family: str, params: Optional[Dict[str, Any]] = None):
        """
        Return an *unfitted* estimator instance given a family string.
        Supported: 'linear' (y ~ t_sec), 'mean' (constant).
        """
        f = (family or "linear").lower()
        params = params or {}
        if f in ("mean", "avg", "baseline"):
            return MeanRegressor()
        if f in ("linear", "ols", "time_linear"):
            # scikit or minimal fallback
            return LinearRegression(**{k: v for k, v in params.items() if k not in {"random_state"}})
        # default: linear
        return LinearRegression()

    # ---------------- training ----------------
    def fit_models(
        self,
        train_df: pd.DataFrame,
        axes: Sequence[str],
        model_config: Optional[ModelConfig] = None,
        *,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fit a per-axis estimator defined by `model_config`.
        Returns dict: {axis_name: fitted_model}
        """
        if "t_sec" not in train_df.columns:
            raise ValueError("Expected column 't_sec' in training DataFrame.")
        rs = self.random_state if random_state is None else int(random_state)

        # Default config if none provided
        if model_config is None:
            model_config = ModelConfig(family="linear", params={})

        self.log.info(
            f"Training models: family='{model_config.family}', params={model_config.params or {}}, "
            f"axes={len(axes)}, random_state={rs}"
        )

        models: Dict[str, Any] = {}
        # Build feature once; sklearn accepts DataFrame or array
        X_full = train_df[["t_sec"]]

        for ax in axes:
            if ax not in train_df.columns:
                self.log.warning(f"Axis '{ax}' missing in TRAIN; model skipped.")
                models[ax] = None
                continue

            # Drop rows with NaNs for the axis or t_sec
            valid = train_df[["t_sec", ax]].dropna()
            if valid.empty or valid[ax].nunique(dropna=True) <= 1:
                self.log.warning(f"Axis '{ax}' has insufficient variance/rows; model skipped.")
                models[ax] = None
                continue

            X = valid[["t_sec"]]
            y = valid[ax].to_numpy()

            est = self._make_estimator(model_config.family, model_config.params)
            est = est.fit(X, y)
            models[ax] = est

        return models

    # (Optional) partial_fit/warm start for streaming or incremental updates
    def partial_fit(
        self,
        models: Dict[str, Any],
        df_chunk: pd.DataFrame,
        axes: Sequence[str],
    ) -> Dict[str, Any]:
        """
        If a model supports .partial_fit, use it; otherwise, leave unchanged.
        """
        if "t_sec" not in df_chunk.columns:
            raise ValueError("Expected column 't_sec' in df_chunk.")
        for ax in axes:
            m = models.get(ax)
            if m is None or ax not in df_chunk.columns:
                continue
            if hasattr(m, "partial_fit"):
                valid = df_chunk[["t_sec", ax]].dropna()
                if valid.empty:
                    continue
                X = valid[["t_sec"]]
                y = valid[ax].to_numpy()
                try:
                    m.partial_fit(X, y)
                except Exception as e:
                    self.log.debug(f"partial_fit not supported for axis '{ax}': {e}")
        return models

    # -------- convenience: build detection dicts --------
    def build_detection_dicts(
        self, models: Dict[str, Any], thresholds: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Convenience structure for detection.
        Returns dict: {axis: {'model': model, 'MinC': .., 'MaxC': .., 'T_sec': ..}}
        """
        out: Dict[str, Any] = {}
        thr_idx = set(thresholds.index.astype(str)) if isinstance(thresholds.index, pd.Index) else set()
        for ax, m in models.items():
            record = {"model": m, "MinC": None, "MaxC": None, "T_sec": None}
            key = str(ax)
            if key in thr_idx:
                row = thresholds.loc[key]
                record["MinC"] = float(row["MinC"]) if "MinC" in thresholds.columns else None
                record["MaxC"] = float(row["MaxC"]) if "MaxC" in thresholds.columns else None
                record["T_sec"] = float(row["T_sec"]) if "T_sec" in thresholds.columns else None
            out[ax] = record
        return out

    # -------- predictions + residuals --------
    def add_preds_residuals(
        self, df: pd.DataFrame, models: Dict[str, Any], axes: Sequence[str]
    ) -> pd.DataFrame:
        """
        Add prediction and residual columns for each axis:
          - pred_{axis}
          - res_{axis} = df[axis] - pred_{axis}
        Returns a new DataFrame with added columns.
        """
        if "t_sec" not in df.columns:
            raise ValueError("Expected column 't_sec' in DataFrame.")
        out = df.copy()

        X = out[["t_sec"]]  # keep as DataFrame for estimators
        for ax in axes:
            col_pred = f"pred_{ax}"
            col_res = f"res_{ax}"

            if ax not in out.columns or models.get(ax) is None:
                out[col_pred] = np.nan
                out[col_res] = np.nan
                continue

            m = models[ax]
            yhat = np.asarray(m.predict(X)).reshape(-1)
            out[col_pred] = yhat
            out[col_res] = out[ax].to_numpy() - yhat

        return out
