from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def create_app(config_path: Path, model_path: Path, use_business_model: bool = False) -> FastAPI:
    """Build and configure FastAPI app exposing model inference endpoints."""
    logger = get_logger("csao.api")

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
        }

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
        pipeline, lite, _ = services.get()
        try:
            frame, meta = lite.build_candidate_frame(
                user_id=req.user_id,
                restaurant_id=req.restaurant_id,
                cart_item_ids=req.cart_item_ids,
                city=req.city,
                hour=req.hour,
                meal_slot=req.meal_slot,
                weather_temp_c=req.weather_temp_c,
                step=req.step,
                candidate_item_ids=req.candidate_item_ids,
                max_candidates=req.max_candidates,
                request_id=req.request_id,
            )
            item_ids = pipeline.recommend(frame, top_k=req.top_k)
        except ValueError as e:
            logger.error("Bad context request payload: %s", e)
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # pragma: no cover - runtime resilience
            logger.exception("Failed to serve context recommend")
            raise HTTPException(status_code=500, detail=f"Internal error: {e}") from e

        return RecommendMainResponse(
            recommended_item_ids=item_ids,
            candidate_count=int(meta["candidate_count"]),
            cold_start_user=bool(meta["cold_start_user"]),
        )

    @app.post("/recommend-main", response_model=RecommendMainResponse)
    def recommend_main(req: RecommendMainRequest) -> RecommendMainResponse:
        return _recommend_with_context(req)

    @app.post("/recommend-lite", response_model=RecommendLiteResponse)
    def recommend_lite(req: RecommendLiteRequest) -> RecommendLiteResponse:
        return _recommend_with_context(req)

    return app

