from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMRanker


@dataclass
class FeatureConfig:
    feature_columns: List[str]
    categorical_features: List[str]
    group_key: Tuple[str, str]
    target: str
    missing_categorical_fill_value: str = "__MISSING__"
    numeric_fill_strategy: str = "median"


class LGBMRankerWrapper:
    """Wrapper around LightGBM ranker with consistent preprocessing."""

    def __init__(self, params: Dict, feature_cfg: FeatureConfig):
        self.params = dict(params)
        self.feature_cfg = feature_cfg
        self.model = LGBMRanker(**self.params)
        self.numeric_fill_values: Dict[str, float] = {}

    @staticmethod
    def _build_groups(df: pd.DataFrame, group_key: Tuple[str, str]) -> np.ndarray:
        grouped = df.groupby(list(group_key), observed=True).size()
        return grouped.to_numpy(dtype=np.int32)

    def _split_feature_types(self) -> Tuple[List[str], List[str]]:
        cat = [c for c in self.feature_cfg.categorical_features if c in self.feature_cfg.feature_columns]
        num = [c for c in self.feature_cfg.feature_columns if c not in set(cat)]
        return cat, num

    def _fit_numeric_fill_values(self, x: pd.DataFrame) -> None:
        _, num_cols = self._split_feature_types()
        if self.feature_cfg.numeric_fill_strategy not in {"median", "zero"}:
            raise ValueError("numeric_fill_strategy must be one of {'median', 'zero'}.")
        for col in num_cols:
            as_num = pd.to_numeric(x[col], errors="coerce")
            if self.feature_cfg.numeric_fill_strategy == "median":
                fill = float(as_num.median()) if not as_num.dropna().empty else 0.0
            else:
                fill = 0.0
            self.numeric_fill_values[col] = fill

    def _transform_features(self, x: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cat_cols, num_cols = self._split_feature_types()
        out = x.copy()

        if fit:
            self._fit_numeric_fill_values(out)

        for col in num_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            fill = self.numeric_fill_values.get(col, 0.0)
            out[col] = out[col].fillna(fill)

        cat_fill = self.feature_cfg.missing_categorical_fill_value
        for col in cat_cols:
            out[col] = out[col].astype("string").fillna(cat_fill)
            out[col] = out[col].astype("category")

        return out

    def _prepare_matrix(
        self,
        df: pd.DataFrame,
        fit: bool,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare transformed feature matrix and LightGBM group sizes."""
        cols = self.feature_cfg.feature_columns
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns in dataframe: {missing}")

        x = df[cols].copy()
        x = self._transform_features(x, fit=fit)

        groups = self._build_groups(df, self.feature_cfg.group_key)
        if groups.sum() != len(df):
            raise ValueError("Inconsistent group sizes: sum(groups) != number of rows.")
        return x, groups

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        early_stopping_rounds: int,
        min_training_rounds: int = 1,
        enable_early_stopping: bool = True,
    ) -> None:
        """Train model with validation early stopping."""
        target_col = self.feature_cfg.target
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in train data.")

        train_df = train_df.sort_values(list(self.feature_cfg.group_key)).reset_index(drop=True)
        val_df = val_df.sort_values(list(self.feature_cfg.group_key)).reset_index(drop=True)

        x_train, g_train = self._prepare_matrix(train_df, fit=True)
        x_val, g_val = self._prepare_matrix(val_df, fit=False)
        y_train = train_df[target_col].to_numpy(dtype=float)
        y_val = val_df[target_col].to_numpy(dtype=float)

        callbacks = [lgb.log_evaluation(period=50)]
        use_early_stopping = enable_early_stopping and int(early_stopping_rounds) > 0
        if use_early_stopping:
            callbacks.insert(0, lgb.early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=True))

        self.model.fit(
            x_train,
            y_train,
            group=g_train,
            eval_set=[(x_val, y_val)],
            eval_group=[g_val],
            eval_names=["val"],
            callbacks=callbacks,
            categorical_feature=self.feature_cfg.categorical_features,
        )

        # Guardrail against pathological early stops that collapse the ranker to 1 tree.
        best_iter = int(getattr(self.model, "best_iteration_", 0) or 0)
        used_rounds = best_iter if best_iter > 0 else int(getattr(self.model, "n_estimators_", 0) or 0)
        min_rounds = max(1, int(min_training_rounds))
        if used_rounds < min_rounds:
            forced_params = dict(self.params)
            forced_params["n_estimators"] = max(min_rounds, int(forced_params.get("n_estimators", min_rounds)))
            self.model = LGBMRanker(**forced_params)
            self.model.fit(
                x_train,
                y_train,
                group=g_train,
                eval_set=[(x_val, y_val)],
                eval_group=[g_val],
                eval_names=["val"],
                callbacks=[lgb.log_evaluation(period=50)],
                categorical_feature=self.feature_cfg.categorical_features,
            )

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores on an already grouped dataframe."""
        df = df.sort_values(list(self.feature_cfg.group_key)).reset_index(drop=True)
        x, _ = self._prepare_matrix(df, fit=False)
        scores = self.model.predict(x)
        return np.asarray(scores, dtype=float)

    def save(self, path: Path) -> None:
        """Persist model and preprocessing artifacts."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": self.params,
            "feature_cfg": self.feature_cfg,
            "model": self.model,
            "numeric_fill_values": self.numeric_fill_values,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "LGBMRankerWrapper":
        """Load a previously saved ranker wrapper."""
        payload = joblib.load(path)
        obj = cls(params=payload["params"], feature_cfg=payload["feature_cfg"])
        obj.model = payload["model"]
        obj.numeric_fill_values = payload.get("numeric_fill_values", {})
        return obj

