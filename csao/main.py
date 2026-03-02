from __future__ import annotations

import errno
import socket
from pathlib import Path

import uvicorn
import yaml

from csao.models.train import train_and_evaluate
from csao.serving.api import create_app
from csao.utils.logger import get_logger


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError as e:
            # EADDRINUSE (Unix) or 10048 / WSAEADDRINUSE (Windows)
            if e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", 10048)):
                return True
            raise


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / "csao" / "config" / "config.yaml").resolve()

    logger = get_logger("csao.main")
    logger.info("Starting training and evaluation pipeline")

    train_and_evaluate(config_path=config_path)

    # Optionally start FastAPI inference server
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("serving", {}).get("enable_api", False):
        model_dir_cfg = Path(cfg["paths"]["model_dir"])
        model_dir = model_dir_cfg if model_dir_cfg.is_absolute() else (project_root / model_dir_cfg).resolve()
        model_path = model_dir / "lgbm_ranker_main.joblib"
        if not model_path.exists():
            logger.error("Model file %s not found; cannot start API.", model_path)
            return

        app = create_app(config_path=config_path, model_path=model_path, use_business_model=False)
        host = cfg["serving"].get("host", "0.0.0.0")
        port = int(cfg["serving"].get("port", 8000))

        if _is_port_in_use(host, port):
            alt_port = port + 1
            logger.warning(
                "Port %d is already in use (another run or app). Using port %d instead. "
                "To free 8000: stop the other process or run 'netstat -ano | findstr :8000' then 'taskkill /PID <pid> /F'.",
                port,
                alt_port,
            )
            port = alt_port

        logger.info("Starting FastAPI server at http://%s:%d", host, port)
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

