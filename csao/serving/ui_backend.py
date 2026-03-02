from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from csao.data.loader import CartSessionsDataLoader
from csao.serving.lite_features import LiteFeatureAssembler
from csao.utils.logger import get_logger


def _mode_string(values: pd.Series, default: str = "") -> str:
    clean = values.dropna().astype(str)
    if clean.empty:
        return default
    mode = clean.mode(dropna=True)
    if mode.empty:
        return default
    return str(mode.iloc[0])


def _median_float(values: pd.Series, default: float = 0.0) -> float:
    num = pd.to_numeric(values, errors="coerce").dropna()
    if num.empty:
        return float(default)
    return float(num.median())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class UiCatalog:
    user_ids: List[str]
    restaurants: List[Dict[str, Any]]
    menu_by_restaurant: Dict[str, List[Dict[str, Any]]]
    combos_by_restaurant: Dict[str, List[Dict[str, Any]]]
    item_name_map_by_restaurant: Dict[str, Dict[str, str]]
    restaurant_id_set: set
    user_id_set: set


class UiBackend:
    """Read-only UI data service built from existing training splits."""

    def __init__(self, config_path: Path, assembler: LiteFeatureAssembler):
        self.logger = get_logger(self.__class__.__name__)
        self.assembler = assembler
        with config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.data_cfg = self.cfg["data"]
        self.catalog = self._load_catalog()

    @property
    def candidate_id_col(self) -> str:
        return self.data_cfg["candidate_id_col"]

    @property
    def restaurant_col(self) -> str:
        return "restaurant_id"

    @property
    def user_col(self) -> str:
        return "user_id"

    def _load_catalog(self) -> UiCatalog:
        loader = CartSessionsDataLoader(self.cfg)
        train_df, val_df, test_df = loader.load_train_val_test()
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        required = {self.user_col, self.restaurant_col, self.candidate_id_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required catalog columns: {sorted(missing)}")

        if "candidate_name" not in df.columns:
            df["candidate_name"] = df[self.candidate_id_col].astype(str)

        df[self.user_col] = df[self.user_col].astype(str)
        df[self.restaurant_col] = df[self.restaurant_col].astype(str)
        df[self.candidate_id_col] = df[self.candidate_id_col].astype(str)

        users = sorted(df[self.user_col].dropna().astype(str).unique().tolist())
        user_set = set(users)

        restaurants = self._build_restaurant_list(df)
        restaurant_set = {r["restaurant_id"] for r in restaurants}

        menu_by_restaurant = self._build_menu(df)
        combos_by_restaurant = self._build_combos(menu_by_restaurant)
        item_name_map_by_restaurant = {
            rid: {item["candidate_item_id"]: item["candidate_name"] for item in items}
            for rid, items in menu_by_restaurant.items()
        }

        self.logger.info(
            "UI catalog ready: users=%d restaurants=%d items=%d",
            len(users),
            len(restaurants),
            sum(len(v) for v in menu_by_restaurant.values()),
        )
        return UiCatalog(
            user_ids=users,
            restaurants=restaurants,
            menu_by_restaurant=menu_by_restaurant,
            combos_by_restaurant=combos_by_restaurant,
            item_name_map_by_restaurant=item_name_map_by_restaurant,
            restaurant_id_set=restaurant_set,
            user_id_set=user_set,
        )

    def _build_restaurant_list(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rest_id, grp in df.groupby(self.restaurant_col, observed=True, sort=True):
            out.append(
                {
                    "restaurant_id": str(rest_id),
                    "city": _mode_string(grp["city"]) if "city" in grp.columns else "",
                    "rest_cuisine": _mode_string(grp["rest_cuisine"]) if "rest_cuisine" in grp.columns else "",
                    "rest_price_tier": _mode_string(grp["rest_price_tier"]) if "rest_price_tier" in grp.columns else "",
                    "rest_rating": round(_median_float(grp["rest_rating"], default=0.0), 2) if "rest_rating" in grp.columns else 0.0,
                    "menu_size": int(grp[self.candidate_id_col].nunique()),
                }
            )
        return out

    def _build_menu(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        menu_by_rest: Dict[str, List[Dict[str, Any]]] = {}

        group_cols = [self.restaurant_col, self.candidate_id_col]
        for (rest_id, item_id), grp in df.groupby(group_cols, observed=True, sort=False):
            rid = str(rest_id)
            row = {
                "candidate_item_id": str(item_id),
                "candidate_name": _mode_string(grp["candidate_name"], default=str(item_id)),
                "candidate_category": _mode_string(grp["candidate_category"], default=""),
                "candidate_cuisine_tag": _mode_string(grp["candidate_cuisine_tag"], default=""),
                "candidate_price": round(_median_float(grp["candidate_price"], default=0.0), 2),
                "candidate_calories": round(_median_float(grp["candidate_calories"], default=0.0), 1),
                "candidate_is_veg": _safe_int(_median_float(grp["candidate_is_veg"], default=0.0), 0),
                "candidate_popularity": round(_median_float(grp["candidate_popularity"], default=0.0), 4),
                "candidate_margin_score": round(_median_float(grp["candidate_margin_score"], default=0.0), 4),
                "pair_max_lift": round(_median_float(grp["pair_max_lift"], default=0.0), 4),
            }
            menu_by_rest.setdefault(rid, []).append(row)

        for rid, items in menu_by_rest.items():
            menu_by_rest[rid] = sorted(
                items,
                key=lambda x: (float(x.get("candidate_popularity", 0.0)), -float(x.get("candidate_price", 0.0))),
                reverse=True,
            )
        return menu_by_rest

    def _build_combos(self, menu_by_restaurant: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        combos_by_rest: Dict[str, List[Dict[str, Any]]] = {}
        for rid, items in menu_by_restaurant.items():
            top_items = items[:12]
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for i in range(len(top_items)):
                for j in range(i + 1, len(top_items)):
                    a, b = top_items[i], top_items[j]
                    if a["candidate_item_id"] == b["candidate_item_id"]:
                        continue
                    a_cat = str(a.get("candidate_category", "")).strip().lower()
                    b_cat = str(b.get("candidate_category", "")).strip().lower()
                    if a_cat and b_cat and a_cat == b_cat:
                        continue
                    score = (
                        float(a.get("candidate_popularity", 0.0))
                        + float(b.get("candidate_popularity", 0.0))
                        + float(a.get("pair_max_lift", 0.0))
                        + float(b.get("pair_max_lift", 0.0))
                    )
                    combo = {
                        "combo_id": f"{a['candidate_item_id']}::{b['candidate_item_id']}",
                        "label": f"{a['candidate_name']} + {b['candidate_name']}",
                        "item_ids": [a["candidate_item_id"], b["candidate_item_id"]],
                        "categories": [a_cat, b_cat],
                        "estimated_price": round(float(a.get("candidate_price", 0.0)) + float(b.get("candidate_price", 0.0)), 2),
                        "combo_score": round(score, 4),
                    }
                    scored.append((score, combo))
            scored.sort(key=lambda x: x[0], reverse=True)
            combos_by_rest[rid] = [c for _, c in scored[:8]]
        return combos_by_rest

    def get_options(self) -> Dict[str, Any]:
        return {
            "users": self.catalog.user_ids,
            "restaurants": self.catalog.restaurants,
        }

    def get_restaurant_menu(self, restaurant_id: str) -> Dict[str, Any]:
        rid = str(restaurant_id)
        if rid not in self.catalog.restaurant_id_set:
            raise ValueError(f"Unknown restaurant_id '{restaurant_id}'")

        restaurant = next((r for r in self.catalog.restaurants if r["restaurant_id"] == rid), None)
        return {
            "restaurant": restaurant,
            "items": self.catalog.menu_by_restaurant.get(rid, []),
            "combos": self.catalog.combos_by_restaurant.get(rid, []),
        }

    def build_candidates(
        self,
        user_id: str,
        restaurant_id: str,
        cart_item_ids: Sequence[str],
        hour: Optional[int] = None,
        meal_slot: Optional[str] = None,
        weather_temp_c: Optional[float] = None,
        step: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> pd.DataFrame:
        uid = str(user_id)
        rid = str(restaurant_id)
        if uid not in self.catalog.user_id_set:
            raise ValueError(f"user_id '{uid}' is not an existing user.")
        if rid not in self.catalog.restaurant_id_set:
            raise ValueError(f"restaurant_id '{rid}' not found.")

        cart_ids = [str(x) for x in cart_item_ids]
        menu_items = self.catalog.menu_by_restaurant.get(rid, [])
        candidate_ids = [item["candidate_item_id"] for item in menu_items if item["candidate_item_id"] not in set(cart_ids)]
        if not candidate_ids:
            raise ValueError("No candidates left after excluding current cart items.")

        restaurant_city = ""
        restaurant = next((r for r in self.catalog.restaurants if r["restaurant_id"] == rid), None)
        if restaurant is not None:
            restaurant_city = str(restaurant.get("city", "")).strip()

        frame, _meta = self.assembler.build_candidate_frame(
            user_id=uid,
            restaurant_id=rid,
            cart_item_ids=cart_ids,
            city=restaurant_city if restaurant_city else None,
            hour=hour,
            meal_slot=meal_slot,
            weather_temp_c=weather_temp_c,
            step=step,
            candidate_item_ids=candidate_ids,
            max_candidates=max(len(candidate_ids), 1),
            request_id=request_id,
        )

        names = self.catalog.item_name_map_by_restaurant.get(rid, {})
        frame["candidate_name"] = frame[self.candidate_id_col].astype(str).map(names).fillna(frame[self.candidate_id_col].astype(str))
        return frame
