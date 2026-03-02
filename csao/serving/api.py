from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from csao.models.inference import InferencePipeline
from csao.serving.lite_features import LiteFeatureAssembler
from csao.serving.ui_backend import UiBackend
from csao.utils.logger import get_logger


class RecommendRequest(BaseModel):
    """Inference request schema."""

    candidates: List[Dict[str, Any]]
    top_k: int = Field(default=5, ge=1, le=50)


class RecommendResponse(BaseModel):
    recommended_item_ids: List[str]


class RecommendMainRequest(BaseModel):
    """Main-model request with minimal UI inputs; backend assembles full feature rows."""

    user_id: str
    restaurant_id: str
    cart_item_ids: List[str] = Field(default_factory=list)
    top_k: int = Field(default=2, ge=1, le=50)
    city: Optional[str] = None
    hour: Optional[int] = Field(default=None, ge=0, le=23)
    meal_slot: Optional[str] = None
    weather_temp_c: Optional[float] = None
    step: Optional[int] = Field(default=None, ge=1)
    candidate_item_ids: Optional[List[str]] = None
    request_id: Optional[str] = None
    max_candidates: int = Field(default=80, ge=1, le=500)


class RecommendMainResponse(BaseModel):
    recommended_item_ids: List[str]
    candidate_count: int
    cold_start_user: bool


class SessionCandidatesRequest(BaseModel):
    user_id: str
    restaurant_id: str
    cart_item_ids: List[str] = Field(default_factory=list)
    hour: Optional[int] = Field(default=None, ge=0, le=23)
    meal_slot: Optional[str] = None
    weather_temp_c: Optional[float] = None
    step: Optional[int] = Field(default=None, ge=1)
    request_id: Optional[str] = None


class SessionCandidatesResponse(BaseModel):
    candidates: List[Dict[str, Any]]
    candidate_count: int


# Backward-compatible aliases for existing clients.
RecommendLiteRequest = RecommendMainRequest
RecommendLiteResponse = RecommendMainResponse


class _TimedResponseCache:
    """Tiny in-process TTL cache for repeated recommendation contexts."""

    def __init__(self, max_entries: int, ttl_sec: float):
        self.max_entries = max(1, int(max_entries))
        self.ttl_sec = max(0.0, float(ttl_sec))
        self._store: "OrderedDict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            hit = self._store.get(key)
            if hit is None:
                return None
            ts, payload = hit
            if now - ts > self.ttl_sec:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return dict(payload)

    def set(self, key: Tuple[Any, ...], payload: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = (time.time(), dict(payload))
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)

    def size(self) -> int:
        with self._lock:
            return len(self._store)


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    dump_fn = getattr(model, "model_dump", None)
    if callable(dump_fn):
        return dump_fn()
    return model.dict()


def _normalize_opt_str(value: Optional[str]) -> str:
    return "" if value is None else str(value)


def _context_cache_key(req: RecommendMainRequest) -> Tuple[Any, ...]:
    cart_ids = tuple(sorted(str(v) for v in req.cart_item_ids))
    explicit_candidates = tuple(str(v) for v in (req.candidate_item_ids or []))
    temp = None if req.weather_temp_c is None else round(float(req.weather_temp_c), 2)
    return (
        str(req.user_id),
        str(req.restaurant_id),
        cart_ids,
        int(req.top_k),
        _normalize_opt_str(req.city),
        int(req.hour) if req.hour is not None else -1,
        _normalize_opt_str(req.meal_slot),
        temp,
        int(req.step) if req.step is not None else -1,
        explicit_candidates,
        int(req.max_candidates),
    )


def create_app(config_path: Path, model_path: Path, use_business_model: bool = False) -> FastAPI:
    """Build and configure FastAPI app exposing model inference endpoints."""
    logger = get_logger("csao.api")

    serving_cfg = {}
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        serving_cfg = cfg.get("serving", {}) or {}
    except Exception:
        serving_cfg = {}

    response_cache: Optional[_TimedResponseCache] = None
    if bool(serving_cfg.get("response_cache_enabled", True)):
        response_cache = _TimedResponseCache(
            max_entries=int(serving_cfg.get("response_cache_max_entries", 2048)),
            ttl_sec=float(serving_cfg.get("response_cache_ttl_sec", 45.0)),
        )

    class _LazyServices:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.pipeline: Optional[InferencePipeline] = None
            self.lite: Optional[LiteFeatureAssembler] = None
            self.ui_backend: Optional[UiBackend] = None
            self._init_error: Optional[Exception] = None

        @property
        def loaded(self) -> bool:
            return self.pipeline is not None and self.lite is not None and self.ui_backend is not None

        def get(self) -> tuple[InferencePipeline, LiteFeatureAssembler, UiBackend]:
            if self.loaded:
                return self.pipeline, self.lite, self.ui_backend  # type: ignore[return-value]
            with self._lock:
                if self.loaded:
                    return self.pipeline, self.lite, self.ui_backend  # type: ignore[return-value]
                if self._init_error is not None:
                    raise self._init_error
                try:
                    pipeline = InferencePipeline(
                        config_path=config_path,
                        model_path=model_path,
                        use_business_model=use_business_model,
                    )
                    lite = LiteFeatureAssembler(config_path=config_path)
                    ui_backend = UiBackend(config_path=config_path, assembler=lite)
                    self.pipeline = pipeline
                    self.lite = lite
                    self.ui_backend = ui_backend
                    return pipeline, lite, ui_backend
                except Exception as e:  # pragma: no cover - startup resilience
                    self._init_error = e
                    raise

    services = _LazyServices()

    app = FastAPI(title="CSAO Cart Add-on Ranker")
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/ui-static", StaticFiles(directory=str(static_dir)), name="ui-static")

    def _ui_response() -> FileResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="UI bundle not found.")
        return FileResponse(index_path)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        lite_ready = bool(services.lite.ready) if services.lite is not None else False
        return {
            "status": "ok",
            "services_loaded": services.loaded,
            "lite_feature_store_ready": lite_ready,
            "response_cache_enabled": response_cache is not None,
            "response_cache_size": response_cache.size() if response_cache is not None else 0,
        }

    @app.on_event("startup")
    def startup_warmup() -> None:
        # Warm heavy services in background so first user action isn't a full cold load.
        if bool(serving_cfg.get("background_warmup_enabled", True)):
            def _warm() -> None:
                try:
                    services.get()
                    logger.info("Background warmup complete")
                except Exception as e:  # pragma: no cover - warmup resilience
                    logger.warning("Background warmup failed: %s", e)

            threading.Thread(target=_warm, daemon=True).start()

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        return _ui_response()

    @app.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        return _ui_response()

    @app.get("/ui/options")
    def ui_options() -> Dict[str, Any]:
        _, _, ui_backend = services.get()
        return ui_backend.get_options()

    @app.get("/ui/restaurants/{restaurant_id}/menu")
    def ui_restaurant_menu(restaurant_id: str) -> Dict[str, Any]:
        _, _, ui_backend = services.get()
        try:
            return ui_backend.get_restaurant_menu(restaurant_id=restaurant_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/ui/session-candidates", response_model=SessionCandidatesResponse)
    def ui_session_candidates(req: SessionCandidatesRequest) -> SessionCandidatesResponse:
        _, _, ui_backend = services.get()
        try:
            frame = ui_backend.build_candidates(
                user_id=req.user_id,
                restaurant_id=req.restaurant_id,
                cart_item_ids=req.cart_item_ids,
                hour=req.hour,
                meal_slot=req.meal_slot,
                weather_temp_c=req.weather_temp_c,
                step=req.step,
                request_id=req.request_id,
            )
        except ValueError as e:
            logger.error("Bad UI candidate request: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        return SessionCandidatesResponse(
            candidates=frame.to_dict(orient="records"),
            candidate_count=int(len(frame)),
        )

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        pipeline, _, _ = services.get()
        if not req.candidates:
            raise HTTPException(status_code=400, detail="No candidates provided.")

        df = pd.DataFrame(req.candidates)
        try:
            item_ids = pipeline.recommend(df, top_k=req.top_k)
        except ValueError as e:
            logger.error("Bad request payload: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e

        return RecommendResponse(recommended_item_ids=item_ids)

    def _recommend_with_context(req: RecommendMainRequest) -> RecommendMainResponse:
        cache_key = _context_cache_key(req)
        if response_cache is not None:
            cached = response_cache.get(cache_key)
            if cached is not None:
                return RecommendMainResponse(**cached)

        pipeline, lite, ui_backend = services.get()
        try:
            cart_set = {str(v) for v in req.cart_item_ids}
            if req.candidate_item_ids:
                explicit_candidates = [str(v) for v in req.candidate_item_ids]
            else:
                # Keep accuracy parity with the original UI path by using the full
                # restaurant menu candidate set rather than a truncated default pool.
                menu_items = ui_backend.catalog.menu_by_restaurant.get(str(req.restaurant_id), [])
                explicit_candidates = [str(item["candidate_item_id"]) for item in menu_items]

            if cart_set:
                explicit_candidates = [cid for cid in explicit_candidates if cid not in cart_set]
            if not explicit_candidates:
                raise ValueError("No candidates left after excluding current cart items.")

            frame, meta = lite.build_candidate_frame(
                user_id=req.user_id,
                restaurant_id=req.restaurant_id,
                cart_item_ids=req.cart_item_ids,
                city=req.city,
                hour=req.hour,
                meal_slot=req.meal_slot,
                weather_temp_c=req.weather_temp_c,
                step=req.step,
                candidate_item_ids=explicit_candidates,
                max_candidates=max(len(explicit_candidates), 1),
                request_id=req.request_id,
            )
            item_ids = pipeline.recommend(frame, top_k=req.top_k)
        except ValueError as e:
            logger.error("Bad context request payload: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # pragma: no cover - runtime resilience
            logger.exception("Failed to serve context recommend")
            raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

        response = RecommendMainResponse(
            recommended_item_ids=item_ids,
            candidate_count=int(meta["candidate_count"]),
            cold_start_user=bool(meta["cold_start_user"]),
        )
        if response_cache is not None:
            response_cache.set(cache_key, _model_dump(response))
        return response

    @app.post("/recommend-main", response_model=RecommendMainResponse)
    def recommend_main(req: RecommendMainRequest) -> RecommendMainResponse:
        return _recommend_with_context(req)

    @app.post("/recommend-lite", response_model=RecommendLiteResponse)
    def recommend_lite(req: RecommendLiteRequest) -> RecommendLiteResponse:
        return _recommend_with_context(req)

    return app

