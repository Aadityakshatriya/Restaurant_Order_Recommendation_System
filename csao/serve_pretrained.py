from __future__ import annotations

import os
from pathlib import Path

import uvicorn
import yaml

from csao.serving.api import create_app
from csao.utils.logger import get_logger


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / "csao" / "config" / "config.yaml").resolve()

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_dir_cfg = Path(cfg["paths"]["model_dir"])
    model_dir = model_dir_cfg if model_dir_cfg.is_absolute() else (project_root / model_dir_cfg).resolve()
    model_path = model_dir / "lgbm_ranker_main.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Main model not found at: {model_path}")

    host = "0.0.0.0"
    # HF Docker Spaces route traffic to container port 7860 unless PORT is injected.
    port = int(os.getenv("PORT", "7860"))

    logger = get_logger("csao.serve_pretrained")
    logger.info("Starting server with pretrained main model: %s", model_path)

    app = create_app(config_path=config_path, model_path=model_path, use_business_model=False)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
