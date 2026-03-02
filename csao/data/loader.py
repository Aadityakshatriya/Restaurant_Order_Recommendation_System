from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from csao.utils.logger import get_logger


@dataclass
class DataPaths:
    cart_sessions_path: Path
    processed_dir: Path


class CartSessionsDataLoader:
    """Data access layer for cart-session ranking data."""

    def __init__(self, cfg: Dict):
        paths_cfg = cfg["paths"]
        data_cfg = cfg["data"]
        self.project_root = Path(__file__).resolve().parents[2]

        self.paths = DataPaths(
            cart_sessions_path=self._resolve_path(paths_cfg["cart_sessions_path"]),
            processed_dir=self._resolve_path(paths_cfg["processed_dir"]),
        )
        self.paths.processed_dir.mkdir(parents=True, exist_ok=True)

        self.split_column: str = data_cfg["split_column"]
        self.expected_splits: List[str] = list(data_cfg["expected_splits"])
        self.group_key = tuple(data_cfg["group_key"])  # (session_id, step)
        self.session_key = data_cfg["session_key"]
        self.required_columns = list(data_cfg.get("required_columns", []))
        self.logger = get_logger(self.__class__.__name__, logs_dir=paths_cfg.get("logs_dir"))

    def _resolve_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @property
    def train_path(self) -> Path:
        return self.paths.processed_dir / "cart_sessions_train.parquet"

    @property
    def val_path(self) -> Path:
        return self.paths.processed_dir / "cart_sessions_val.parquet"

    @property
    def test_path(self) -> Path:
        return self.paths.processed_dir / "cart_sessions_test.parquet"

    def _validate_schema(self, df: pd.DataFrame, source_name: str) -> None:
        required = set(self.required_columns + list(self.group_key))
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{source_name} is missing required columns: {sorted(missing)}")

    def _validate_split_integrity(self, df: pd.DataFrame) -> None:
        if self.split_column not in df.columns:
            raise ValueError(f"Expected '{self.split_column}' column in raw dataset.")

        found_splits = set(df[self.split_column].dropna().unique().tolist())
        missing_splits = set(self.expected_splits) - found_splits
        if missing_splits:
            raise ValueError(f"Missing expected splits in data: {sorted(missing_splits)}")

        # No leakage across sessions.
        session_split_counts = df.groupby(self.session_key, observed=True)[self.split_column].nunique()
        if (session_split_counts > 1).any():
            raise ValueError("Data leakage detected: some session_id values appear in multiple splits.")

        # No leakage across ranking groups.
        group_split_counts = df.groupby(list(self.group_key), observed=True)[self.split_column].nunique()
        if (group_split_counts > 1).any():
            raise ValueError("Data leakage detected: some (session_id, step) groups appear in multiple splits.")

    def ensure_splits(self) -> None:
        """Create split files from raw data when needed."""
        if self.train_path.exists() and self.val_path.exists() and self.test_path.exists():
            self.logger.info("Using existing split parquet files in %s", self.paths.processed_dir)
            return

        if not self.paths.cart_sessions_path.exists():
            raise FileNotFoundError(f"Raw cart_sessions file not found: {self.paths.cart_sessions_path}")

        self.logger.info("Loading raw cart_sessions from %s", self.paths.cart_sessions_path)
        df = pd.read_parquet(self.paths.cart_sessions_path)
        self._validate_schema(df, source_name="Raw cart_sessions parquet")
        self._validate_split_integrity(df)

        self.logger.info("Writing split parquet files to %s", self.paths.processed_dir)
        for split_name, path in [("train", self.train_path), ("val", self.val_path), ("test", self.test_path)]:
            sub = df[df[self.split_column] == split_name].copy()
            if sub.empty:
                raise ValueError(f"No rows found for split='{split_name}'.")
            self.logger.info("Split '%s': %d rows", split_name, len(sub))
            sub.to_parquet(path, index=False)

    def load_split(self, split: str) -> pd.DataFrame:
        """
        Load a pre-split parquet file.

        Parameters
        ----------
        split:
            One of 'train', 'val', 'test'.
        """
        mapping: Dict[str, Path] = {
            "train": self.train_path,
            "val": self.val_path,
            "test": self.test_path,
        }
        if split not in mapping:
            raise ValueError(f"Unknown split '{split}'. Expected one of {list(mapping.keys())}")
        path = mapping[split]
        if not path.exists():
            self.ensure_splits()
        if not path.exists():
            raise FileNotFoundError(f"Split parquet not found for split='{split}' at {path}")
        self.logger.info("Loading %s split from %s", split, path)
        df = pd.read_parquet(path)
        self._validate_schema(df, source_name=f"{split} split parquet")
        return df

    def load_train_val_test(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation, and test dataframes."""
        self.ensure_splits()
        return (
            self.load_split("train"),
            self.load_split("val"),
            self.load_split("test"),
        )

