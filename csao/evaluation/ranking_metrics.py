from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def build_group_sizes(df: pd.DataFrame, group_cols: Sequence[str]) -> np.ndarray:
    """Build LightGBM-compatible group sizes from ranking keys."""
    return df.groupby(list(group_cols), observed=True).size().to_numpy(dtype=np.int32)


def _iter_groups(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray):
    """Yield grouped labels and scores based on group sizes."""
    assert labels.shape == scores.shape
    assert groups.sum() == len(labels)
    offset = 0
    for g in groups:
        g_int = int(g)
        y = labels[offset : offset + g_int]
        s = scores[offset : offset + g_int]
        offset += g_int
        yield y, s


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted cumulative gain at K."""
    r = np.asarray(relevances, dtype=float)[:k]
    if r.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, r.size + 2))
    return float(np.sum(r / discounts))


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, groups: np.ndarray, k: int) -> float:
    """Compute mean NDCG@k across ranking groups."""
    ndcgs: List[float] = []
    for y, s in _iter_groups(y_true, scores, groups):
        order = np.argsort(-s)
        y_sorted = y[order]
        dcg = dcg_at_k(y_sorted, k)
        ideal = dcg_at_k(np.sort(y)[::-1], k)
        if ideal > 0:
            ndcgs.append(dcg / ideal)
    if not ndcgs:
        return 0.0
    return float(np.mean(ndcgs))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, groups: np.ndarray, k: int) -> float:
    """Compute mean Precision@k across ranking groups."""
    precisions: List[float] = []
    for y, s in _iter_groups(y_true, scores, groups):
        if len(y) == 0:
            continue
        order = np.argsort(-s)
        top = y[order][:k]
        precisions.append(float(top.sum()) / float(k))
    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, groups: np.ndarray, k: int) -> float:
    """Compute mean Recall@k across ranking groups."""
    recalls: List[float] = []
    for y, s in _iter_groups(y_true, scores, groups):
        total_pos = float(y.sum())
        if total_pos <= 0:
            continue
        order = np.argsort(-s)
        top = y[order][:k]
        recalls.append(float(top.sum()) / total_pos)
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def auc_overall(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute global ROC-AUC over all rows."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def coverage_at_k(
    candidate_ids: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    k: int,
) -> float:
    """Compute unique item coverage at K."""
    assert candidate_ids.shape == scores.shape
    n_rows = len(candidate_ids)
    if n_rows == 0:
        return 0.0

    all_unique = set(candidate_ids.tolist())
    rec_unique: set = set()
    offset = 0
    for g in groups:
        g_int = int(g)
        ids = candidate_ids[offset : offset + g_int]
        s = scores[offset : offset + g_int]
        offset += g_int
        if g_int == 0:
            continue
        order = np.argsort(-s)
        top_ids = ids[order][:k]
        rec_unique.update(top_ids.tolist())
    if not all_unique:
        return 0.0
    return float(len(rec_unique) / len(all_unique))


def evaluate_ranking(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    candidate_id_col: str,
    group_cols: Sequence[str],
    top_k: int,
) -> Dict[str, float]:
    """Compute core ranking metrics for a scored dataframe."""
    data = df.sort_values(list(group_cols)).reset_index(drop=True)
    y_true = data[target_col].to_numpy(dtype=float)
    scores = data[score_col].to_numpy(dtype=float)
    groups = build_group_sizes(data, group_cols)
    return {
        f"ndcg@{top_k}": ndcg_at_k(y_true=y_true, scores=scores, groups=groups, k=top_k),
        f"precision@{top_k}": precision_at_k(y_true=y_true, scores=scores, groups=groups, k=top_k),
        f"recall@{top_k}": recall_at_k(y_true=y_true, scores=scores, groups=groups, k=top_k),
        "auc": auc_overall(y_true=y_true, scores=scores),
        f"coverage@{top_k}": coverage_at_k(
            candidate_ids=data[candidate_id_col].to_numpy(),
            scores=scores,
            groups=groups,
            k=top_k,
        ),
    }


def evaluate_ranking_by_step(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    candidate_id_col: str,
    group_cols: Sequence[str],
    step_col: str,
    top_k: int,
) -> Dict[int, Dict[str, float]]:
    """Compute ranking metrics separately for each session step."""
    out: Dict[int, Dict[str, float]] = {}
    for step, sub in df.groupby(step_col, observed=True):
        out[int(step)] = evaluate_ranking(
            df=sub,
            score_col=score_col,
            target_col=target_col,
            candidate_id_col=candidate_id_col,
            group_cols=group_cols,
            top_k=top_k,
        )
    return out


def expected_aov_lift(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    aov_lift_col: str,
    group_cols: Sequence[str],
    top_k: int,
) -> float:
    """Estimate expected AOV lift using top-K picked true additions."""
    data = df.sort_values(list(group_cols)).reset_index(drop=True)
    y_true = data[target_col].to_numpy(dtype=float)
    scores = data[score_col].to_numpy(dtype=float)
    aov_lift = data[aov_lift_col].to_numpy(dtype=float)
    groups = build_group_sizes(data, group_cols)

    total_lift = 0.0
    offset = 0
    for g in groups:
        g_int = int(g)
        y = y_true[offset : offset + g_int]
        s = scores[offset : offset + g_int]
        lift = aov_lift[offset : offset + g_int]
        offset += g_int

        order = np.argsort(-s)
        top_idx = order[:top_k]
        total_lift += float(np.sum(y[top_idx] * lift[top_idx]))

    n_groups = int(len(groups))
    return total_lift / n_groups if n_groups > 0 else 0.0

