# ModelSelection/modelselection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional: normality / distribution tests if SciPy is available
try:
    from scipy import stats as spstats  # type: ignore
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ============================================================
# Simple, dependency-light estimators (time-based)
# Each estimator supports: .fit(X[['t_sec']], y) and .predict(X[['t_sec']])
# ============================================================
class MeanRegressor:
    def __init__(self):
        self.mu_: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MeanRegressor":
        self.mu_ = float(np.nanmean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.mu_ is None:
            raise RuntimeError("Model not fitted")
        return np.full(shape=(len(X),), fill_value=self.mu_, dtype=float)


class LinearTimeRegressor:
    """
    y ≈ a * t_sec + b  (OLS via np.polyfit)
    """
    def __init__(self):
        self.coef_: float | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LinearTimeRegressor":
        t = np.asarray(X["t_sec"], dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        # Handle degenerate cases
        if len(t) < 2 or np.allclose(t, t[0]):
            self.coef_, self.intercept_ = 0.0, float(np.nanmean(y))
        else:
            a, b = np.polyfit(t, y, deg=1)
            self.coef_, self.intercept_ = float(a), float(b)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model not fitted")
        t = np.asarray(X["t_sec"], dtype=float).reshape(-1)
        return self.coef_ * t + self.intercept_


# ============================================================
# Strategy config
# ============================================================
@dataclass
class ModelConfig:
    """
    Example:
      ModelConfig(family="linear", params={})
      ModelConfig(family="mean", params={})
    """
    family: str
    params: Dict[str, Any] | None = None


class ModelSelector:
    """
    Align test stats and produce similarity reports + simple strategy/factory & selector.
    Cells: 3.0c, 3.0d, 4.1 (+ new: build_models, score_selector)
    """

    # -------- 3.0c --------
    def align_test_to_train_stats(
        self,
        test: pd.DataFrame,
        train: pd.DataFrame,
        axes: Sequence[str],
    ) -> pd.DataFrame:
        """
        Affine alignment per axis (per-column):
            x' = (x - mu_te) / sd_te * sd_tr + mu_tr

        If sd_te == 0 (constant), only shift to mu_tr.
        Non-axis columns are copied as-is.

        POLICY: TRAIN is the **reference distribution** for residual/threshold
        parity across train/evaluate/live (unless explicitly disabled).
        This keeps residuals comparable to TRAIN-derived thresholds.
        """
        out = test.copy()
        for ax in axes:
            if ax not in test.columns or ax not in train.columns:
                continue
            s_tr = pd.to_numeric(train[ax], errors="coerce").dropna()
            s_te = pd.to_numeric(test[ax], errors="coerce")

            if s_tr.empty:
                continue

            mu_tr, sd_tr = float(s_tr.mean()), float(s_tr.std(ddof=1))
            mu_te = float(s_te.mean(skipna=True))
            sd_te = float(s_te.std(ddof=1, skipna=True))

            if sd_te and np.isfinite(sd_te) and sd_te > 0:
                out[ax] = ((s_te - mu_te) / sd_te) * (sd_tr if np.isfinite(sd_tr) else 1.0) + mu_tr
            else:
                out[ax] = s_te - mu_te + mu_tr

        return out

    # -------- 3.0d --------
    def similarity_report(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        axes: Sequence[str],
    ) -> pd.DataFrame:
        """
        Per-axis summary: means, stds, deltas, ratios, min/max.
        If SciPy available: add Shapiro-Wilk p-values for normality per split.
        """
        rows: list[Dict[str, Any]] = []
        for ax in axes:
            if ax not in train.columns or ax not in test.columns:
                rows.append({
                    "axis": ax, "μ_train": np.nan, "σ_train": np.nan,
                    "μ_test": np.nan, "σ_test": np.nan,
                    "Δμ": np.nan, "Δμ_%": np.nan, "σ_ratio": np.nan,
                    "min_train": np.nan, "max_train": np.nan,
                    "min_test": np.nan, "max_test": np.nan,
                    "p_shapiro_train": np.nan, "p_shapiro_test": np.nan,
                })
                continue

            tr = pd.to_numeric(train[ax], errors="coerce").dropna()
            te = pd.to_numeric(test[ax], errors="coerce").dropna()

            mu_tr = float(tr.mean()) if not tr.empty else np.nan
            sd_tr = float(tr.std(ddof=1)) if len(tr) > 1 else np.nan
            mu_te = float(te.mean()) if not te.empty else np.nan
            sd_te = float(te.std(ddof=1)) if len(te) > 1 else np.nan

            dmu = (mu_te - mu_tr) if np.isfinite(mu_tr) and np.isfinite(mu_te) else np.nan
            dmu_pct = (dmu / abs(mu_tr) * 100.0) if np.isfinite(dmu) and mu_tr not in (0.0, np.nan) else np.nan
            sratio = (sd_te / sd_tr) if (np.isfinite(sd_tr) and sd_tr not in (0.0, np.nan) and np.isfinite(sd_te)) else np.nan

            p_tr = p_te = np.nan
            if _HAVE_SCIPY:
                try:
                    if len(tr) >= 3:
                        p_tr = float(spstats.shapiro(tr).pvalue)
                    if len(te) >= 3:
                        p_te = float(spstats.shapiro(te).pvalue)
                except Exception:
                    p_tr = p_te = np.nan

            rows.append({
                "axis": ax,
                "μ_train": mu_tr, "σ_train": sd_tr,
                "μ_test": mu_te,  "σ_test": sd_te,
                "Δμ": dmu, "Δμ_%": dmu_pct,
                "σ_ratio": sratio,
                "min_train": float(tr.min()) if not tr.empty else np.nan,
                "max_train": float(tr.max()) if not tr.empty else np.nan,
                "min_test": float(te.min()) if not te.empty else np.nan,
                "max_test": float(te.max()) if not te.empty else np.nan,
                "p_shapiro_train": p_tr,
                "p_shapiro_test": p_te,
            })

        rep = pd.DataFrame(rows)
        cols = [
            "axis",
            "μ_train", "σ_train", "μ_test", "σ_test",
            "Δμ", "Δμ_%", "σ_ratio",
            "min_train", "max_train", "min_test", "max_test",
            "p_shapiro_train", "p_shapiro_test",
        ]
        return rep[cols]

    # -------- 4.1 --------
    def compare_test_candidates(
        self,
        candidates: Dict[str, pd.DataFrame],
        train: pd.DataFrame,
        axes: Sequence[str],
    ) -> pd.DataFrame:
        """
        Compare multiple TEST candidates vs TRAIN using a simple score:
          score = mean(|Δμ_%|) + mean(|log(σ_ratio)|)  (lower is better)
        Returns a DataFrame sorted by score ascending.
        """
        rows = []
        for name, df in candidates.items():
            rep = self.similarity_report(train, df, axes)
            dmu_pct = pd.to_numeric(rep["Δμ_%"], errors="coerce").dropna()
            sratio = pd.to_numeric(rep["σ_ratio"], errors="coerce").dropna()

            miss_penalty = (len(rep) - max(len(dmu_pct), len(sratio))) * 5.0
            comp1 = float(dmu_pct.abs().mean()) if not dmu_pct.empty else np.inf
            comp2 = float(np.abs(np.log(sratio.replace({0: np.nan}))).mean()) if not sratio.empty else np.inf
            score = comp1 + comp2 + miss_penalty

            rows.append({
                "candidate": name,
                "mean_abs_Δμ_%": comp1,
                "mean_abs_log_σ_ratio": comp2,
                "missing_penalty": miss_penalty,
                "score": score,
            })

        out = pd.DataFrame(rows).sort_values(by="score", ascending=True).reset_index(drop=True)
        return out

    # ============================================================
    # NEW: Strategy factory + simple selector (prep for Lab 2)
    # ============================================================
    def _make_estimator(self, family: str, params: Optional[Dict[str, Any]] = None):
        """
        Return an *unfitted* estimator instance given a family string.
        """
        f = (family or "").lower()
        params = params or {}
        if f in ("mean", "avg", "baseline"):
            return MeanRegressor()
        if f in ("linear", "ols", "time_linear"):
            return LinearTimeRegressor()
        # Fallback to linear as a sensible default
        return LinearTimeRegressor()

    def build_models(
        self,
        axes: Sequence[str],
        config: ModelConfig,
        *,
        prefit: bool = False,
        train_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Factory: build per-axis estimators according to `config`.
        If prefit=True, also fits each model on `train_df[['t_sec']]` vs `train_df[axis]`.
        Returns dict {axis: estimator}.
        """
        models: Dict[str, Any] = {}
        for ax in axes:
            est = self._make_estimator(config.family, config.params)
            models[ax] = est

        if prefit:
            if train_df is None:
                raise ValueError("prefit=True requires train_df")
            X = train_df[["t_sec"]] if "t_sec" in train_df.columns else self._ensure_time_feature(train_df)
            for ax in axes:
                if ax in train_df.columns:
                    y = np.asarray(train_df[ax], dtype=float).reshape(-1)
                    models[ax].fit(X, y)
        return models

    def score_selector(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        axes: Sequence[str],
        candidates: List[ModelConfig],
        *,
        metric: str = "mae",
    ) -> Tuple[ModelConfig, pd.DataFrame]:
        """
        Fit & score each candidate config (simple holdout using provided train/test).
        Returns (best_config, leaderboard_df)
        leaderboard columns: ['family','params','score','per_axis_mae']
        """
        if metric.lower() != "mae":
            raise ValueError("Only 'mae' metric is supported in this lightweight selector.")

        leaderboard = []
        Xtr = train[["t_sec"]] if "t_sec" in train.columns else self._ensure_time_feature(train)
        Xte = test[["t_sec"]] if "t_sec" in test.columns else self._ensure_time_feature(test)

        for cfg in candidates:
            models = self.build_models(axes, cfg, prefit=True, train_df=train)
            per_axis_mae = {}
            maes = []
            for ax in axes:
                if ax not in test.columns:
                    continue
                y_true = np.asarray(test[ax], dtype=float).reshape(-1)
                y_pred = np.asarray(models[ax].predict(Xte)).reshape(-1)
                mae = float(np.nanmean(np.abs(y_true - y_pred)))
                per_axis_mae[ax] = mae
                maes.append(mae)
            score = float(np.nanmean(maes)) if maes else np.inf
            leaderboard.append({
                "family": cfg.family,
                "params": cfg.params or {},
                "score": score,
                "per_axis_mae": per_axis_mae,
            })

        lb_df = pd.DataFrame(leaderboard).sort_values("score", ascending=True).reset_index(drop=True)
        if lb_df.empty:
            # Fall back to a trivial config
            best = ModelConfig(family="linear", params={})
            return best, lb_df
        best_row = lb_df.iloc[0]
        best = ModelConfig(family=str(best_row["family"]), params=dict(best_row["params"]))
        return best, lb_df

    # -------- NEW: axes picker used by CLI --------
    def pick_axes(self, df: pd.DataFrame, top_k: int = 8) -> list[str]:
        """Choose top_k numeric axes (exclude Time, t_sec, id, Trait)."""
        ban = {"Time", "t_sec", "Trait", "id"}   # exclude metadata columns
        num = df.select_dtypes(include=[np.number]).columns
        cand = [c for c in num if c not in ban]
        if not cand:
            return []
        var = df[cand].var(numeric_only=True).sort_values(ascending=False)
        return list(var.index[:top_k])

    # ---------------- Internal helper ----------------
    def _ensure_time_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        if "t_sec" in df.columns:
            return df[["t_sec"]]
        # best-effort from 'Time'
        if "Time" in df.columns:
            t = pd.to_datetime(df["Time"])
            tsec = (t - t.iloc[0]).dt.total_seconds().astype(float)
            return pd.DataFrame({"t_sec": tsec}, index=df.index)
        raise ValueError("Expected 't_sec' or 'Time' to build time feature.")
