from __future__ import annotations

import itertools
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from csao.data.loader import CartSessionsDataLoader
from csao.evaluation.ranking_metrics import evaluate_ranking, evaluate_ranking_by_step, expected_aov_lift
from csao.models.lgbm_ranker import FeatureConfig, LGBMRankerWrapper
from csao.utils.logger import get_logger


@dataclass
class TrainingOutput:
    model_name: str
    model_path: Path
    metrics: Dict[str, float]
    metrics_by_step: Dict[int, Dict[str, float]]
    expected_aov_lift: float
    best_params: Dict[str, float]
    score_strategy: str


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    model_dir: Path
    registry_dir: Path


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_path(project_root: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _setup_run_artifacts(cfg: Dict, project_root: Path) -> RunArtifacts:
    model_dir = _resolve_path(project_root, cfg["paths"]["model_dir"])
    registry_dir = _resolve_path(project_root, cfg["paths"]["registry_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = registry_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunArtifacts(run_id=run_id, run_dir=run_dir, model_dir=model_dir, registry_dir=registry_dir)


def _validate_feature_schema(df: pd.DataFrame, cfg: Dict) -> None:
    data_cfg = cfg["data"]
    required = set(data_cfg.get("required_columns", []))
    features = set(data_cfg["feature_columns"])
    categorical = set(data_cfg["categorical_features"])
    missing = (required | features) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    if not categorical.issubset(features):
        raise ValueError("Every categorical feature must be listed in feature_columns.")


def _build_feature_config(cfg: Dict, target_col: str) -> FeatureConfig:
    data_cfg = cfg["data"]
    return FeatureConfig(
        feature_columns=list(data_cfg["feature_columns"]),
        categorical_features=list(data_cfg["categorical_features"]),
        group_key=tuple(data_cfg["group_key"]),  # type: ignore[arg-type]
        target=target_col,
        missing_categorical_fill_value=data_cfg["missing"]["categorical_fill_value"],
        numeric_fill_strategy=data_cfg["missing"]["numeric_fill_strategy"],
    )


def _coerce_training_types(df: pd.DataFrame, feature_cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cfg.categorical_features:
        out[col] = out[col].astype("string")
    return out


def _bin_business_label(train_values: np.ndarray, val_values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_bins < 2:
        raise ValueError("business_label_bins must be >= 2")
    q = np.linspace(0.0, 1.0, n_bins + 1)
    cuts = np.quantile(train_values, q)
    cuts = np.unique(cuts)
    if cuts.size <= 2:
        cuts = np.linspace(np.min(train_values), np.max(train_values), n_bins + 1)
    boundaries = cuts[1:-1]
    train_int = np.digitize(train_values, boundaries, right=True)
    val_int = np.digitize(val_values, boundaries, right=True)
    return train_int.astype(int), val_int.astype(int)


def _minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if np.isclose(lo, hi):
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _build_final_score(df: pd.DataFrame, model_score_col: str, cfg: Dict) -> np.ndarray:
    rerank_cfg = cfg.get("rerank", {})
    if not rerank_cfg.get("enabled", False):
        return df[model_score_col].to_numpy(dtype=float)
    wm = float(rerank_cfg.get("blend_weight_model", 0.7))
    wc = float(rerank_cfg.get("blend_weight_margin", 0.3))
    margin = pd.to_numeric(df["candidate_margin_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model_score = df[model_score_col].to_numpy(dtype=float)
    return wm * _minmax(model_score) + wc * _minmax(margin)


def _build_blended_score(df: pd.DataFrame, model_score_col: str, cfg: Dict) -> np.ndarray:
    rerank_cfg = cfg.get("rerank", {})
    wm = float(rerank_cfg.get("blend_weight_model", 0.7))
    wc = float(rerank_cfg.get("blend_weight_margin", 0.3))
    margin = pd.to_numeric(df["candidate_margin_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    model_score = df[model_score_col].to_numpy(dtype=float)
    return wm * _minmax(model_score) + wc * _minmax(margin)


def _cart_state_diversity_at_1(
    df: pd.DataFrame,
    score_col: str,
    candidate_id_col: str,
    group_cols: List[str],
    user_col: str = "user_id",
    restaurant_col: str = "restaurant_id",
) -> float:
    """Mean per-(user,restaurant) top-1 diversity across decision contexts."""
    required = {score_col, candidate_id_col, user_col, restaurant_col, *group_cols}
    if not required.issubset(df.columns):
        return 0.0

    sort_cols = list(group_cols) + [score_col]
    ascending = [True] * len(group_cols) + [False]
    ranked = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    top1 = ranked.groupby(group_cols, observed=True, sort=False).head(1).copy()
    if top1.empty:
        return 0.0

    ratios: List[float] = []
    for _, sub in top1.groupby([user_col, restaurant_col], observed=True, sort=False):
        n = int(len(sub))
        if n < 2:
            continue
        ratios.append(float(sub[candidate_id_col].astype(str).nunique()) / float(n))
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def _evaluate_scored_df(df: pd.DataFrame, score_col: str, cfg: Dict) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], float]:
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    top_k = int(eval_cfg["top_k"])
    metrics = evaluate_ranking(
        df=df,
        score_col=score_col,
        target_col=data_cfg["target"],
        candidate_id_col=data_cfg["candidate_id_col"],
        group_cols=data_cfg["group_key"],
        top_k=top_k,
    )
    metrics["cart_state_diversity@1"] = _cart_state_diversity_at_1(
        df=df,
        score_col=score_col,
        candidate_id_col=data_cfg["candidate_id_col"],
        group_cols=list(data_cfg["group_key"]),
        user_col="user_id",
        restaurant_col="restaurant_id",
    )
    if eval_cfg.get("compute_step_metrics", True):
        by_step = evaluate_ranking_by_step(
            df=df,
            score_col=score_col,
            target_col=data_cfg["target"],
            candidate_id_col=data_cfg["candidate_id_col"],
            group_cols=data_cfg["group_key"],
            step_col=data_cfg["step_col"],
            top_k=top_k,
        )
    else:
        by_step = {}
    aov = expected_aov_lift(
        df=df,
        score_col=score_col,
        target_col=data_cfg["target"],
        aov_lift_col=data_cfg["aov_lift_col"],
        group_cols=data_cfg["group_key"],
        top_k=top_k,
    )
    return metrics, by_step, aov


def _sample_groups(df: pd.DataFrame, group_cols: List[str], frac: float, seed: int) -> pd.DataFrame:
    """Sample ranking groups for faster/exhaustive tuning while preserving group integrity."""
    if frac >= 1.0:
        return df
    if frac <= 0.0:
        raise ValueError("Group sample fraction must be > 0.")

    group_df = df[group_cols].drop_duplicates().reset_index(drop=True)
    if group_df.empty:
        return df
    n_groups = int(max(1, round(len(group_df) * frac)))
    sampled = group_df.sample(n=n_groups, random_state=seed, replace=False)
    keep_idx = pd.MultiIndex.from_frame(sampled[group_cols])
    group_idx = pd.MultiIndex.from_frame(df[group_cols])
    mask = group_idx.isin(keep_idx)
    out = df.loc[mask].copy()
    return out.reset_index(drop=True)


def _tuning_objective_score(metrics: Dict[str, float], cfg: Dict) -> float:
    tuning_cfg = cfg.get("tuning", {})
    weighted = tuning_cfg.get("objective_weights", {})
    if weighted:
        score = 0.0
        for key, weight in weighted.items():
            if key not in metrics:
                continue
            score += float(weight) * float(metrics[key])
        return float(score)
    objective_name = tuning_cfg["optimize_for"]
    return float(metrics[objective_name])


def _score_on_validation(
    ranker: LGBMRankerWrapper,
    val_df: pd.DataFrame,
    model_name: str,
    cfg: Dict,
) -> Tuple[float, Dict[str, float]]:
    val_sorted = val_df.sort_values(cfg["data"]["group_key"]).reset_index(drop=True).copy()
    score_col = f"score_{model_name}"
    val_sorted[score_col] = ranker.predict_scores(val_sorted)
    val_sorted[f"{score_col}_final"] = _build_final_score(val_sorted, score_col, cfg)
    metrics = evaluate_ranking(
        df=val_sorted,
        score_col=f"{score_col}_final",
        target_col=cfg["data"]["target"],
        candidate_id_col=cfg["data"]["candidate_id_col"],
        group_cols=cfg["data"]["group_key"],
        top_k=int(cfg["evaluation"]["top_k"]),
    )
    metrics["cart_state_diversity@1"] = _cart_state_diversity_at_1(
        df=val_sorted,
        score_col=f"{score_col}_final",
        candidate_id_col=cfg["data"]["candidate_id_col"],
        group_cols=list(cfg["data"]["group_key"]),
        user_col="user_id",
        restaurant_col="restaurant_id",
    )
    score = _tuning_objective_score(metrics=metrics, cfg=cfg)
    return score, metrics


def _select_score_strategy(
    ranker: LGBMRankerWrapper,
    val_df: pd.DataFrame,
    model_name: str,
    cfg: Dict,
    logger,
) -> str:
    rerank_cfg = cfg.get("rerank", {})
    if not rerank_cfg.get("enabled", False):
        return "raw"

    val_sorted = val_df.sort_values(cfg["data"]["group_key"]).reset_index(drop=True).copy()
    score_col = f"score_{model_name}"
    val_sorted[score_col] = ranker.predict_scores(val_sorted)
    val_sorted[f"{score_col}_blend"] = _build_blended_score(val_sorted, score_col, cfg)
    objective_name = cfg["tuning"]["optimize_for"]

    raw_metrics = evaluate_ranking(
        df=val_sorted,
        score_col=score_col,
        target_col=cfg["data"]["target"],
        candidate_id_col=cfg["data"]["candidate_id_col"],
        group_cols=cfg["data"]["group_key"],
        top_k=int(cfg["evaluation"]["top_k"]),
    )
    blend_metrics = evaluate_ranking(
        df=val_sorted,
        score_col=f"{score_col}_blend",
        target_col=cfg["data"]["target"],
        candidate_id_col=cfg["data"]["candidate_id_col"],
        group_cols=cfg["data"]["group_key"],
        top_k=int(cfg["evaluation"]["top_k"]),
    )
    raw_score = float(raw_metrics[objective_name])
    blend_score = float(blend_metrics[objective_name])
    if rerank_cfg.get("apply_only_if_better", True) and blend_score < raw_score:
        logger.warning(
            "Disabled rerank for %s because blended validation %s=%.6f < raw %.6f",
            model_name,
            objective_name,
            blend_score,
            raw_score,
        )
        return "raw"
    return "blend"


def _generate_param_trials(cfg: Dict) -> List[Dict]:
    base = dict(cfg["model"]["lgbm_params"])
    tuning_cfg = cfg.get("tuning", {})
    if not tuning_cfg.get("enabled", False):
        return [base]

    search = tuning_cfg.get("search_space", {})
    keys = sorted(search.keys())
    if not keys:
        return [base]

    values = [list(search[k]) for k in keys]
    combos = list(itertools.product(*values))
    exhaustive = bool(tuning_cfg.get("exhaustive", False))
    max_trials = len(combos) if exhaustive else int(tuning_cfg.get("max_trials", len(combos)))
    trials: List[Dict] = []
    for combo in combos[:max_trials]:
        params = dict(base)
        for k, v in zip(keys, combo):
            params[k] = v
        trials.append(params)
    return trials


def _tune_params(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    cfg: Dict,
    logger,
) -> Dict:
    trials = _generate_param_trials(cfg)
    if len(trials) == 1:
        params = dict(trials[0])
        params["random_state"] = int(cfg["seed"])
        return params

    tuning_cfg = cfg.get("tuning", {})
    train_frac = float(tuning_cfg.get("train_group_sample_frac", 1.0))
    val_frac = float(tuning_cfg.get("val_group_sample_frac", 1.0))
    group_cols = list(cfg["data"]["group_key"])
    train_tune = _sample_groups(train_df, group_cols=group_cols, frac=train_frac, seed=int(cfg["seed"]) + 11)
    val_tune = _sample_groups(val_df, group_cols=group_cols, frac=val_frac, seed=int(cfg["seed"]) + 29)
    logger.info(
        "Tuning dataset sampled: train=%d/%d rows, val=%d/%d rows",
        len(train_tune),
        len(train_df),
        len(val_tune),
        len(val_df),
    )

    best_params: Optional[Dict] = None
    best_score = -np.inf
    for i, params in enumerate(trials, start=1):
        p = dict(params)
        p["random_state"] = int(cfg["seed"])
        logger.info("Tuning %s: trial %d/%d params=%s", model_name, i, len(trials), p)
        ranker = LGBMRankerWrapper(params=p, feature_cfg=feature_cfg)
        ranker.fit(
            train_df=train_tune,
            val_df=val_tune,
            early_stopping_rounds=int(cfg["training"]["early_stopping_rounds"]),
            min_training_rounds=int(cfg["training"].get("min_training_rounds", 1)),
            enable_early_stopping=bool(cfg["training"].get("enable_early_stopping", True)),
        )
        score, metrics = _score_on_validation(ranker=ranker, val_df=val_tune, model_name=model_name, cfg=cfg)
        logger.info(
            "Tuning %s: trial %d score=%0.6f ndcg@5=%0.6f recall@5=%0.6f coverage@5=%0.6f cart_state_diversity@1=%0.6f",
            model_name,
            i,
            score,
            float(metrics.get("ndcg@5", float("nan"))),
            float(metrics.get("recall@5", float("nan"))),
            float(metrics.get("coverage@5", float("nan"))),
            float(metrics.get("cart_state_diversity@1", float("nan"))),
        )
        if score > best_score:
            best_score = score
            best_params = p

    assert best_params is not None
    logger.info(
        "Tuning %s complete. Best %s=%0.6f params=%s",
        model_name,
        cfg["tuning"]["optimize_for"],
        best_score,
        best_params,
    )
    return best_params


def _save_model_artifacts(
    ranker: LGBMRankerWrapper,
    model_name: str,
    model_dir: Path,
    run_dir: Path,
    save_feature_importance: bool,
) -> Tuple[Path, Optional[Path]]:
    latest_model_path = model_dir / f"lgbm_ranker_{model_name}.joblib"
    run_model_path = run_dir / f"lgbm_ranker_{model_name}.joblib"
    ranker.save(latest_model_path)
    shutil.copy2(latest_model_path, run_model_path)

    imp_path = None
    if save_feature_importance:
        imp = pd.DataFrame(
            {
                "feature": ranker.feature_cfg.feature_columns,
                "importance": ranker.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        latest_imp = model_dir / f"feature_importances_{model_name}.csv"
        run_imp = run_dir / f"feature_importances_{model_name}.csv"
        imp.to_csv(latest_imp, index=False)
        shutil.copy2(latest_imp, run_imp)
        imp_path = latest_imp
    return latest_model_path, imp_path


def _train_single_ranker(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cfg: FeatureConfig,
    cfg: Dict,
    artifacts: RunArtifacts,
    logger,
) -> TrainingOutput:
    best_params = _tune_params(
        model_name=model_name,
        train_df=train_df,
        val_df=val_df,
        feature_cfg=feature_cfg,
        cfg=cfg,
        logger=logger,
    )
    ranker = LGBMRankerWrapper(params=best_params, feature_cfg=feature_cfg)
    ranker.fit(
        train_df=train_df,
        val_df=val_df,
        early_stopping_rounds=int(cfg["training"]["early_stopping_rounds"]),
        min_training_rounds=int(cfg["training"].get("min_training_rounds", 1)),
        enable_early_stopping=bool(cfg["training"].get("enable_early_stopping", True)),
    )

    score_strategy = _select_score_strategy(ranker, val_df, model_name, cfg, logger)
    test_sorted = test_df.sort_values(cfg["data"]["group_key"]).reset_index(drop=True)
    test_sorted[f"score_{model_name}"] = ranker.predict_scores(test_sorted)
    if score_strategy == "blend":
        test_sorted[f"score_{model_name}_final"] = _build_blended_score(test_sorted, f"score_{model_name}", cfg)
    else:
        test_sorted[f"score_{model_name}_final"] = test_sorted[f"score_{model_name}"].to_numpy(dtype=float)
    metrics, metrics_by_step, aov_lift = _evaluate_scored_df(
        df=test_sorted,
        score_col=f"score_{model_name}_final",
        cfg=cfg,
    )

    model_path, imp_path = _save_model_artifacts(
        ranker=ranker,
        model_name=model_name,
        model_dir=artifacts.model_dir,
        run_dir=artifacts.run_dir,
        save_feature_importance=bool(cfg["training"].get("save_feature_importance", True)),
    )
    logger.info("Saved %s model to %s", model_name, model_path)
    if imp_path is not None:
        logger.info("Saved %s feature importances to %s", model_name, imp_path)

    return TrainingOutput(
        model_name=model_name,
        model_path=model_path,
        metrics=metrics,
        metrics_by_step=metrics_by_step,
        expected_aov_lift=aov_lift,
        best_params=best_params,
        score_strategy=score_strategy,
    )


def train_and_evaluate(config_path: Path) -> None:
    """Train and evaluate contextual ranking models and save artifacts."""
    cfg = _load_config(config_path)
    project_root = Path(__file__).resolve().parents[2]
    artifacts = _setup_run_artifacts(cfg, project_root=project_root)
    logs_dir = cfg["paths"].get("logs_dir")
    logger = get_logger("csao.train", logs_dir=logs_dir)

    _set_seed(int(cfg["seed"]))
    logger.info("Loaded config from %s", config_path)
    logger.info("Run ID: %s", artifacts.run_id)

    loader = CartSessionsDataLoader(cfg)
    train_df, val_df, test_df = loader.load_train_val_test()
    _validate_feature_schema(train_df, cfg)
    _validate_feature_schema(val_df, cfg)
    _validate_feature_schema(test_df, cfg)

    main_target = cfg["training"]["main_target_name"]
    business_target = cfg["training"]["business_target_name"]

    main_feature_cfg = _build_feature_config(cfg, target_col=main_target)
    train_main = _coerce_training_types(train_df, main_feature_cfg)
    val_main = _coerce_training_types(val_df, main_feature_cfg)
    test_main = _coerce_training_types(test_df, main_feature_cfg)
    train_out_main = _train_single_ranker(
        model_name="main",
        train_df=train_main,
        val_df=val_main,
        test_df=test_main,
        feature_cfg=main_feature_cfg,
        cfg=cfg,
        artifacts=artifacts,
        logger=logger,
    )

    train_business_model = bool(cfg["training"].get("train_business_model", False))
    train_out_business: Optional[TrainingOutput] = None
    if train_business_model:
        bins = int(cfg["training"]["business_label_bins"])
        train_business = train_df.copy()
        val_business = val_df.copy()
        business_train_int, business_val_int = _bin_business_label(
            train_values=train_business[business_target].to_numpy(dtype=float),
            val_values=val_business[business_target].to_numpy(dtype=float),
            n_bins=bins,
        )
        business_int_col = f"{business_target}_int"
        train_business[business_int_col] = business_train_int
        val_business[business_int_col] = business_val_int
        test_business = test_df.copy()
        test_business[business_int_col] = 0

        business_feature_cfg = _build_feature_config(cfg, target_col=business_int_col)
        train_business = _coerce_training_types(train_business, business_feature_cfg)
        val_business = _coerce_training_types(val_business, business_feature_cfg)
        test_business = _coerce_training_types(test_business, business_feature_cfg)
        train_out_business = _train_single_ranker(
            model_name="business",
            train_df=train_business,
            val_df=val_business,
            test_df=test_business,
            feature_cfg=business_feature_cfg,
            cfg=cfg,
            artifacts=artifacts,
            logger=logger,
        )
    else:
        logger.info("Skipping business model training (training.train_business_model=false)")

    baseline_df = test_df.sort_values(cfg["data"]["group_key"]).reset_index(drop=True).copy()
    baseline_df["score_baseline"] = baseline_df[cfg["data"]["popularity_col"]].to_numpy(dtype=float)
    baseline_metrics, baseline_by_step, baseline_aov = _evaluate_scored_df(
        df=baseline_df,
        score_col="score_baseline",
        cfg=cfg,
    )

    logger.info("Baseline metrics: %s", baseline_metrics)
    logger.info("Main model metrics: %s", train_out_main.metrics)
    if train_out_business is not None:
        logger.info("Business model metrics: %s", train_out_business.metrics)
        logger.info(
            "Expected AOV lift - baseline=%.6f | main=%.6f | business=%.6f",
            baseline_aov,
            train_out_main.expected_aov_lift,
            train_out_business.expected_aov_lift,
        )
    else:
        logger.info(
            "Expected AOV lift - baseline=%.6f | main=%.6f",
            baseline_aov,
            train_out_main.expected_aov_lift,
        )

    if baseline_aov != 0.0:
        main_uplift_pct = ((train_out_main.expected_aov_lift - baseline_aov) / baseline_aov) * 100.0
        business_uplift_pct = (
            ((train_out_business.expected_aov_lift - baseline_aov) / baseline_aov) * 100.0
            if train_out_business is not None
            else None
        )
    else:
        main_uplift_pct = float("nan")
        business_uplift_pct = float("nan") if train_out_business is not None else None
    if train_out_business is not None:
        logger.info("Projected AOV uplift vs baseline: main=%.3f%% | business=%.3f%%", main_uplift_pct, business_uplift_pct)
    else:
        logger.info("Projected AOV uplift vs baseline: main=%.3f%%", main_uplift_pct)

    metrics_payload = {
        "run_id": artifacts.run_id,
        "baseline": {
            "metrics": baseline_metrics,
            "metrics_by_step": baseline_by_step,
            "expected_aov_lift": baseline_aov,
        },
        "main": {
            "metrics": train_out_main.metrics,
            "metrics_by_step": train_out_main.metrics_by_step,
            "expected_aov_lift": train_out_main.expected_aov_lift,
            "model_path": str(train_out_main.model_path),
            "best_params": train_out_main.best_params,
            "score_strategy": train_out_main.score_strategy,
        },
        "uplift_vs_baseline_pct": {
            "main": main_uplift_pct,
        },
    }
    if train_out_business is not None:
        metrics_payload["business"] = {
            "metrics": train_out_business.metrics,
            "metrics_by_step": train_out_business.metrics_by_step,
            "expected_aov_lift": train_out_business.expected_aov_lift,
            "model_path": str(train_out_business.model_path),
            "best_params": train_out_business.best_params,
            "score_strategy": train_out_business.score_strategy,
        }
        metrics_payload["uplift_vs_baseline_pct"]["business"] = business_uplift_pct

    latest_metrics_path = artifacts.model_dir / "metrics_summary.json"
    with latest_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    run_metrics_path = artifacts.run_dir / "metrics_summary.json"
    with run_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    manifest = {
        "latest_run_id": artifacts.run_id,
        "latest_metrics_path": str(latest_metrics_path),
        "latest_main_model_path": str(artifacts.model_dir / "lgbm_ranker_main.joblib"),
        "registry_dir": str(artifacts.registry_dir),
    }
    business_model_path = artifacts.model_dir / "lgbm_ranker_business.joblib"
    if business_model_path.exists():
        manifest["latest_business_model_path"] = str(business_model_path)
    with (artifacts.registry_dir / "latest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved latest metrics to %s", latest_metrics_path)
    logger.info("Saved run metrics to %s", run_metrics_path)
    logger.info("Updated registry manifest at %s", artifacts.registry_dir / "latest.json")
