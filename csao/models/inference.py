from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from csao.models.lgbm_ranker import FeatureConfig, LGBMRankerWrapper
from csao.utils.logger import get_logger


class InferencePipeline:
    """Inference pipeline for contextual cart add-on recommendation."""

    def __init__(self, config_path: Path, model_path: Path, use_business_model: bool = False):
        self.config_path = config_path
        self.model_path = model_path
        self.logger = get_logger(self.__class__.__name__)

        with config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        data_cfg = self.cfg["data"]
        group_key = tuple(data_cfg["group_key"])
        self.model = LGBMRankerWrapper.load(model_path)

        target = self.cfg["training"]["business_target_name"] if use_business_model else self.cfg["training"]["main_target_name"]
        fc = self.model.feature_cfg
        self.feature_cfg = FeatureConfig(
            feature_columns=fc.feature_columns,
            categorical_features=fc.categorical_features,
            group_key=group_key,  # type: ignore[arg-type]
            target=target,
            missing_categorical_fill_value=fc.missing_categorical_fill_value,
            numeric_fill_strategy=fc.numeric_fill_strategy,
        )
        self.group_key = list(group_key)
        self.candidate_id_col = data_cfg["candidate_id_col"]
        self.has_dessert_col = data_cfg["has_dessert_col"]
        self.category_col = data_cfg["candidate_category_col"]
        self.rerank_cfg = self.cfg.get("rerank", {})

        self.logger.info("Loaded model from %s", model_path)
        self.logger.info("Using %d features for inference", len(self.feature_cfg.feature_columns))

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if np.isclose(lo, hi):
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    def _apply_optional_rerank(self, df: pd.DataFrame, model_scores: np.ndarray) -> np.ndarray:
        if not self.rerank_cfg.get("enabled", False):
            return model_scores

        wm = float(self.rerank_cfg.get("blend_weight_model", 0.7))
        wc = float(self.rerank_cfg.get("blend_weight_margin", 0.3))
        margin = pd.to_numeric(df.get("candidate_margin_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return wm * self._minmax(model_scores) + wc * self._minmax(margin)

    def recommend(self, session_df: pd.DataFrame, top_k: int = 5) -> List[str]:
        """Return top-k candidate item ids for a single (session_id, step)."""
        required = set(self.feature_cfg.feature_columns) | {self.candidate_id_col} | set(self.feature_cfg.group_key)
        missing = required - set(session_df.columns)
        if missing:
            raise ValueError(f"Missing columns for inference: {missing}")

        # Must represent exactly one ranking context.
        group_counts = session_df.groupby(self.group_key, observed=True).ngroups
        if group_counts != 1:
            raise ValueError("recommend() expects exactly one (session_id, step) group per request.")

        data = session_df.copy()
        scores = self.model.predict_scores(data)
        final_scores = self._apply_optional_rerank(data, scores)
        order = np.argsort(-scores)
        ranked = data.iloc[order].copy().reset_index(drop=True)
        ranked["final_score"] = final_scores[order]
        ranked = ranked.sort_values("final_score", ascending=False).reset_index(drop=True)

        # Optional business rule: if dessert already in cart, avoid >1 dessert recommendation.
        if self.rerank_cfg.get("dessert_guard_enabled", False):
            has_dessert = int(pd.to_numeric(ranked[self.has_dessert_col], errors="coerce").fillna(0).max())
            if has_dessert == 1 and self.category_col in ranked.columns:
                selected_rows: List[int] = []
                dessert_picked = 0
                for idx, row in ranked.iterrows():
                    is_dessert = str(row[self.category_col]).strip().lower() == "dessert"
                    if is_dessert and dessert_picked >= 1:
                        continue
                    selected_rows.append(idx)
                    if is_dessert:
                        dessert_picked += 1
                    if len(selected_rows) >= top_k:
                        break
                top = ranked.iloc[selected_rows]
            else:
                top = ranked.head(top_k)
        else:
            top = ranked.head(top_k)

        return top[self.candidate_id_col].astype(str).tolist()

