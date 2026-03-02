from __future__ import annotations

import ast
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from csao.data.loader import CartSessionsDataLoader
from csao.utils.logger import get_logger


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(pd.isna(value))


def _as_str(value: Any) -> str:
    return str(value) if value is not None else ""


class LiteFeatureAssembler:
    """Build full model feature rows from minimal online request context."""

    _CATEGORY_KEYWORDS = {
        "main": ("main", "entree"),
        "side": ("side", "addon"),
        "drink": ("drink", "beverage", "juice", "soda", "coffee", "tea"),
        "dessert": ("dessert", "sweet", "cake", "ice cream"),
        "snack": ("snack", "starter", "appetizer"),
    }

    def __init__(self, config_path: Path):
        self.logger = get_logger(self.__class__.__name__)
        self.config_path = config_path
        with config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        data_cfg = self.cfg["data"]
        self.feature_columns: List[str] = list(data_cfg["feature_columns"])
        self.categorical_features = set(data_cfg["categorical_features"])
        self.group_key: List[str] = list(data_cfg["group_key"])
        self.session_col = self.group_key[0]
        self.step_col = self.group_key[1]

        self.user_col = "user_id"
        self.restaurant_col = "restaurant_id"
        self.candidate_col = data_cfg["candidate_id_col"]
        self.missing_cat = data_cfg["missing"]["categorical_fill_value"]
        self.popularity_col = data_cfg.get("popularity_col", "candidate_popularity")

        self.user_feature_cols = [c for c in self.feature_columns if c.startswith("user_")]
        self.rest_feature_cols = [c for c in self.feature_columns if c.startswith("rest_")]
        self.candidate_feature_cols = [c for c in self.feature_columns if c.startswith("candidate_")]

        self.global_numeric_defaults: Dict[str, float] = {}
        self.global_categorical_defaults: Dict[str, str] = {}
        self.global_candidate_ids: List[str] = []
        self.seen_meal_slots: set = set()

        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.restaurant_profiles: Dict[str, Dict[str, Any]] = {}
        self.restaurant_candidate_ids: Dict[str, List[str]] = {}
        self.item_profiles: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.global_item_profiles: Dict[str, Dict[str, Any]] = {}
        self.item_price: Dict[str, float] = {}
        self.item_category: Dict[str, str] = {}
        self.item_pair_lift: Dict[str, float] = {}
        self.item_pair_seen: Dict[str, float] = {}
        self.pair_positive_counts: Dict[Tuple[str, str], int] = {}
        self.candidate_positive_totals: Dict[str, int] = {}

        self.ready = False
        self.bootstrap_error: Optional[str] = None
        self._bootstrap()

    def _bootstrap(self) -> None:
        try:
            loader = CartSessionsDataLoader(self.cfg)
            train_df, val_df, test_df = loader.load_train_val_test()
            df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            self._validate_reference_df(df)
            self._compute_global_defaults(df)
            self._build_profiles(df)
            self.ready = True
            self.logger.info(
                "Lite feature assembler ready: users=%d restaurants=%d restaurant-candidates=%d",
                len(self.user_profiles),
                len(self.restaurant_profiles),
                len(self.item_profiles),
            )
        except Exception as e:  # pragma: no cover - resilience path
            self.bootstrap_error = str(e)
            self.ready = False
            self.logger.exception("Failed to bootstrap lite feature assembler: %s", e)

    def _validate_reference_df(self, df: pd.DataFrame) -> None:
        required = {self.user_col, self.restaurant_col, self.candidate_col, self.session_col, self.step_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required online reference columns: {sorted(missing)}")

    def _numeric_default(self, feature: str) -> float:
        return float(self.global_numeric_defaults.get(feature, 0.0))

    def _categorical_default(self, feature: str) -> str:
        return self.global_categorical_defaults.get(feature, self.missing_cat)

    def _aggregate_series(self, s: pd.Series, feature: str) -> Any:
        if feature in self.categorical_features:
            clean = s.dropna().astype(str)
            if clean.empty:
                return self._categorical_default(feature)
            mode = clean.mode(dropna=True)
            return str(mode.iloc[0]) if not mode.empty else self._categorical_default(feature)

        as_num = pd.to_numeric(s, errors="coerce").dropna()
        if as_num.empty:
            return self._numeric_default(feature)
        return float(as_num.median())

    def _compute_global_defaults(self, df: pd.DataFrame) -> None:
        for feature in self.feature_columns:
            if feature not in df.columns:
                if feature in self.categorical_features:
                    self.global_categorical_defaults[feature] = self.missing_cat
                else:
                    self.global_numeric_defaults[feature] = 0.0
                continue

            if feature in self.categorical_features:
                self.global_categorical_defaults[feature] = _as_str(self._aggregate_series(df[feature], feature))
            else:
                self.global_numeric_defaults[feature] = float(self._aggregate_series(df[feature], feature))

        self.seen_meal_slots = set(df["meal_slot"].dropna().astype(str).unique().tolist()) if "meal_slot" in df.columns else set()

        if self.popularity_col in df.columns:
            ranked = (
                df.groupby(self.candidate_col, observed=True)[self.popularity_col]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
        else:
            ranked = df[self.candidate_col].astype(str).value_counts().index.tolist()
        self.global_candidate_ids = [str(v) for v in ranked]

    def _aggregate_profiles(self, df: pd.DataFrame, key_col: str, profile_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        cols = [c for c in profile_cols if c in df.columns]
        if key_col not in df.columns or not cols:
            return out

        sub = df[[key_col] + cols].copy()
        sub[key_col] = sub[key_col].astype(str)
        for key, grp in sub.groupby(key_col, observed=True, sort=False):
            row: Dict[str, Any] = {}
            for feature in cols:
                row[feature] = self._aggregate_series(grp[feature], feature)
            out[str(key)] = row
        return out

    def _build_profiles(self, df: pd.DataFrame) -> None:
        base = df.copy()
        for col in [self.user_col, self.restaurant_col, self.candidate_col]:
            base[col] = base[col].astype(str)

        self.user_profiles = self._aggregate_profiles(
            base,
            key_col=self.user_col,
            profile_cols=self.user_feature_cols + ["city"],
        )
        self.restaurant_profiles = self._aggregate_profiles(
            base,
            key_col=self.restaurant_col,
            profile_cols=self.rest_feature_cols + ["city"],
        )

        item_profile_cols = list(dict.fromkeys(self.candidate_feature_cols + ["aov_lift_if_added"]))
        for (restaurant_id, item_id), grp in base.groupby([self.restaurant_col, self.candidate_col], observed=True, sort=False):
            key = (str(restaurant_id), str(item_id))
            row: Dict[str, Any] = {}
            for feature in item_profile_cols:
                if feature in grp.columns:
                    row[feature] = self._aggregate_series(grp[feature], feature)
            self.item_profiles[key] = row

        for item_id, grp in base.groupby(self.candidate_col, observed=True, sort=False):
            row = {}
            for feature in item_profile_cols:
                if feature in grp.columns:
                    row[feature] = self._aggregate_series(grp[feature], feature)
            item_key = str(item_id)
            self.global_item_profiles[item_key] = row
            if "candidate_price" in grp.columns:
                price = pd.to_numeric(grp["candidate_price"], errors="coerce").median()
                self.item_price[item_key] = 0.0 if pd.isna(price) else float(price)
            else:
                self.item_price[item_key] = self._numeric_default("candidate_price")

            if "candidate_category" in grp.columns:
                category = self._aggregate_series(grp["candidate_category"], "candidate_category")
                self.item_category[item_key] = _as_str(category).strip().lower()
            else:
                self.item_category[item_key] = ""
            if "pair_max_lift" in grp.columns:
                self.item_pair_lift[item_key] = float(pd.to_numeric(grp["pair_max_lift"], errors="coerce").median())
            if "pair_seen_before_flag" in grp.columns:
                self.item_pair_seen[item_key] = float(pd.to_numeric(grp["pair_seen_before_flag"], errors="coerce").median())

        if self.popularity_col in base.columns:
            rank_df = (
                base.groupby([self.restaurant_col, self.candidate_col], observed=True)[self.popularity_col]
                .median()
                .reset_index()
                .sort_values([self.restaurant_col, self.popularity_col], ascending=[True, False])
            )
        else:
            rank_df = (
                base.groupby([self.restaurant_col, self.candidate_col], observed=True)
                .size()
                .reset_index(name="count")
                .sort_values([self.restaurant_col, "count"], ascending=[True, False])
            )

        for restaurant_id, grp in rank_df.groupby(self.restaurant_col, observed=True, sort=False):
            self.restaurant_candidate_ids[str(restaurant_id)] = grp[self.candidate_col].astype(str).tolist()

        self._build_pair_affinity(base)

    def _parse_cart_item_ids(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, tuple):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return []
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed if str(v).strip()]
            except Exception:
                pass
            if txt.startswith("[") and txt.endswith("]"):
                txt = txt[1:-1]
            parts = [p.strip().strip("'").strip('"') for p in txt.split(",")]
            return [p for p in parts if p]
        return []

    def _build_pair_affinity(self, df: pd.DataFrame) -> None:
        if "cart_item_ids" not in df.columns or "added" not in df.columns:
            return

        sub = df[[self.candidate_col, "cart_item_ids", "added"]].copy()
        sub[self.candidate_col] = sub[self.candidate_col].astype(str)
        sub["added"] = pd.to_numeric(sub["added"], errors="coerce").fillna(0.0)
        pos = sub[sub["added"] > 0.5]
        if pos.empty:
            return

        pair_counts: Dict[Tuple[str, str], int] = {}
        cand_totals: Dict[str, int] = {}
        for row in pos.itertuples(index=False):
            cand = str(getattr(row, self.candidate_col))
            cand_totals[cand] = cand_totals.get(cand, 0) + 1
            cart_ids = self._parse_cart_item_ids(getattr(row, "cart_item_ids"))
            if not cart_ids:
                continue
            for cart_item_id in set(cart_ids):
                key = (cart_item_id, cand)
                pair_counts[key] = pair_counts.get(key, 0) + 1

        self.pair_positive_counts = pair_counts
        self.candidate_positive_totals = cand_totals

    def _infer_meal_slot(self, hour: int) -> str:
        if 5 <= hour <= 10:
            slot = "breakfast"
        elif 11 <= hour <= 15:
            slot = "lunch"
        elif 16 <= hour <= 18:
            slot = "snack"
        elif 19 <= hour <= 22:
            slot = "dinner"
        else:
            slot = "late_night"
        if self.seen_meal_slots and slot not in self.seen_meal_slots:
            return self._categorical_default("meal_slot")
        return slot

    def _contains_any(self, text: str, tokens: Tuple[str, ...]) -> bool:
        lower = text.strip().lower()
        return any(t in lower for t in tokens)

    def _cart_flags(self, cart_item_ids: List[str]) -> Dict[str, int]:
        flags = {"has_main": 0, "has_side": 0, "has_drink": 0, "has_dessert": 0, "has_snack": 0}
        for item_id in cart_item_ids:
            cat = self.item_category.get(item_id, "")
            if self._contains_any(cat, self._CATEGORY_KEYWORDS["main"]):
                flags["has_main"] = 1
            if self._contains_any(cat, self._CATEGORY_KEYWORDS["side"]):
                flags["has_side"] = 1
            if self._contains_any(cat, self._CATEGORY_KEYWORDS["drink"]):
                flags["has_drink"] = 1
            if self._contains_any(cat, self._CATEGORY_KEYWORDS["dessert"]):
                flags["has_dessert"] = 1
            if self._contains_any(cat, self._CATEGORY_KEYWORDS["snack"]):
                flags["has_snack"] = 1
        return flags

    def _select_candidates(
        self,
        restaurant_id: str,
        explicit_candidates: Optional[List[str]],
        max_candidates: int,
    ) -> List[str]:
        if explicit_candidates:
            return [str(v) for v in explicit_candidates][:max_candidates]
        restaurant_candidates = self.restaurant_candidate_ids.get(restaurant_id, [])
        if restaurant_candidates:
            return restaurant_candidates[:max_candidates]
        return self.global_candidate_ids[:max_candidates]

    def _build_base_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for feature in self.feature_columns:
            if feature in self.categorical_features:
                row[feature] = self._categorical_default(feature)
            else:
                row[feature] = self._numeric_default(feature)
        return row

    def _safe_num(self, value: Any, default: float) -> float:
        out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(out):
            return float(default)
        return float(out)

    def build_candidate_frame(
        self,
        user_id: str,
        restaurant_id: str,
        cart_item_ids: Optional[List[str]] = None,
        city: Optional[str] = None,
        hour: Optional[int] = None,
        meal_slot: Optional[str] = None,
        weather_temp_c: Optional[float] = None,
        step: Optional[int] = None,
        candidate_item_ids: Optional[List[str]] = None,
        max_candidates: int = 80,
        request_id: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if (not self.ready) and (not candidate_item_ids):
            detail = self.bootstrap_error or "reference store unavailable"
            raise ValueError(
                "Lite feature store is not warmed. Provide candidate_item_ids or load reference data first. "
                f"detail={detail}"
            )

        user_id = str(user_id)
        restaurant_id = str(restaurant_id)
        cart_ids = [str(v) for v in (cart_item_ids or [])]
        now_hour = int(datetime.now().hour)
        req_hour = int(hour) if hour is not None else now_hour
        req_meal_slot = (meal_slot or self._infer_meal_slot(req_hour)).strip()

        candidates = self._select_candidates(
            restaurant_id=restaurant_id,
            explicit_candidates=[str(v) for v in candidate_item_ids] if candidate_item_ids else None,
            max_candidates=max_candidates,
        )
        if not candidates:
            raise ValueError(
                "No candidates available for this restaurant. Provide candidate_item_ids explicitly or warm the reference store."
            )

        user_profile = self.user_profiles.get(user_id, {})
        rest_profile = self.restaurant_profiles.get(restaurant_id, {})
        cold_start_user = user_id not in self.user_profiles

        req_city = city or _as_str(user_profile.get("city")) or _as_str(rest_profile.get("city"))
        if not req_city:
            req_city = self._categorical_default("city")
        temp_c = self._safe_num(weather_temp_c, self._numeric_default("weather_temp_c"))
        is_hot = 1 if temp_c >= 30.0 else 0

        cart_prices = [self.item_price[i] for i in cart_ids if i in self.item_price]
        cart_size = len(cart_ids)
        if cart_prices:
            cart_total = float(sum(cart_prices))
            avg_cart_price = cart_total / len(cart_prices)
        else:
            cart_total = float(cart_size) * self._numeric_default("candidate_price")
            avg_cart_price = self._numeric_default("candidate_price")
        if cart_size == 0:
            cart_total = 0.0
            avg_cart_price = self._numeric_default("candidate_price")

        req_step = int(step) if step is not None else max(1, cart_size + 1)
        cart_flags = self._cart_flags(cart_ids)

        base_common = self._build_base_row()
        base_common["step"] = req_step
        base_common["hour"] = req_hour
        base_common["city"] = req_city
        base_common["meal_slot"] = req_meal_slot
        base_common["cart_total"] = cart_total
        base_common["cart_size"] = cart_size
        base_common["cart_momentum"] = min(1.0, float(cart_size) / 4.0)
        base_common["weather_temp_c"] = temp_c
        base_common["is_hot_weather"] = is_hot
        for k, v in cart_flags.items():
            base_common[k] = int(v)

        for feature in self.user_feature_cols:
            if feature in user_profile and not _is_missing(user_profile[feature]):
                base_common[feature] = user_profile[feature]
        for feature in self.rest_feature_cols:
            if feature in rest_profile and not _is_missing(rest_profile[feature]):
                base_common[feature] = rest_profile[feature]

        base_common["is_cold_start"] = 1 if cold_start_user else int(self._safe_num(base_common.get("is_cold_start"), 0.0))
        if cold_start_user:
            base_common["user_ordered_before"] = 0
            base_common["user_orders_30d"] = 0.0
            base_common["user_orders_7d"] = 0.0

        session_id = request_id or f"lite::{user_id}::{restaurant_id}::{int(time.time() * 1000)}"

        rows: List[Dict[str, Any]] = []
        for candidate_id in candidates:
            row = dict(base_common)
            row[self.session_col] = session_id
            row[self.step_col] = req_step
            row[self.user_col] = user_id
            row[self.restaurant_col] = restaurant_id
            row[self.candidate_col] = candidate_id

            cand_profile = self.item_profiles.get((restaurant_id, candidate_id)) or self.global_item_profiles.get(candidate_id, {})
            for feature in self.candidate_feature_cols + ["aov_lift_if_added"]:
                if feature in cand_profile and not _is_missing(cand_profile[feature]):
                    row[feature] = cand_profile[feature]

            candidate_price = self._safe_num(row.get("candidate_price"), self._numeric_default("candidate_price"))
            user_ref_price = self._safe_num(
                row.get("user_median_item_price_90d"),
                avg_cart_price if avg_cart_price > 0 else self._numeric_default("candidate_price"),
            )
            if user_ref_price <= 0:
                user_ref_price = self._numeric_default("candidate_price")
            row["price_delta_cart_avg"] = candidate_price - (avg_cart_price if avg_cart_price > 0 else user_ref_price)

            if _is_missing(row.get("candidate_in_price_sweet_spot")):
                tolerance = max(1.0, 0.35 * user_ref_price)
                row["candidate_in_price_sweet_spot"] = 1 if abs(candidate_price - user_ref_price) <= tolerance else 0

            category = _as_str(row.get("candidate_category")).strip().lower()
            row["weather_drink_affinity"] = 1.0 if (is_hot == 1 and self._contains_any(category, self._CATEGORY_KEYWORDS["drink"])) else 0.0

            default_pair_lift = self._numeric_default("pair_max_lift")
            default_pair_seen = self._numeric_default("pair_seen_before_flag")
            if cart_size > 0:
                cand_total = int(self.candidate_positive_totals.get(candidate_id, 0))
                pair_scores: List[float] = []
                pair_seen = 0.0
                if cand_total > 0:
                    for cart_item_id in set(cart_ids):
                        pair_pos = int(self.pair_positive_counts.get((cart_item_id, candidate_id), 0))
                        if pair_pos > 0:
                            pair_seen = 1.0
                            pair_scores.append(float(pair_pos) / float(cand_total))
                row["pair_max_lift"] = max(pair_scores) if pair_scores else self.item_pair_lift.get(candidate_id, default_pair_lift)
                row["pair_seen_before_flag"] = pair_seen if pair_scores else self.item_pair_seen.get(candidate_id, default_pair_seen)
            else:
                row["pair_max_lift"] = 0.0
                row["pair_seen_before_flag"] = 0.0

            for feature in self.feature_columns:
                if feature in row and not _is_missing(row[feature]):
                    continue
                if feature in self.categorical_features:
                    row[feature] = self._categorical_default(feature)
                else:
                    row[feature] = self._numeric_default(feature)

            rows.append(row)

        assembled = pd.DataFrame(rows)
        meta = {
            "cold_start_user": cold_start_user,
            "candidate_count": len(rows),
            "restaurant_known": restaurant_id in self.restaurant_profiles,
        }
        return assembled, meta
