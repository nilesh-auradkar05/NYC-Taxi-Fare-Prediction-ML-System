"""
Common helpers for local training and batch inference runners.
"""

from __future__ import annotations

import logging
import logging.config
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import yaml

DEFAULT_PROJECT_CONFIG = "config/local.yml"
DEFAULT_DATASET = "Dataset/yellow_tripdata_2025-09.parquet"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def generate_run_id() -> str:
    return f"{int(time.time() * 1000)}-{uuid4().hex[:8]}"


def _expand_environment(value: Any) -> Any:
    pattern = re.compile(r"\$\{(\w+)\}")

    if isinstance(value, str):
        return pattern.sub(lambda match: os.getenv(match.group(1), match.group(0)), value)
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    return value


def parse_project_configuration(contents: str | None) -> dict[str, Any]:
    """
    Parse project configuration and expand values like `${ENVIRONMENT_VARIABLE}`.
    """
    config = yaml.safe_load(contents or "") or {}
    if not isinstance(config, dict):
        raise ValueError("Project configuration must be a mapping")

    config = _expand_environment(config)
    config.setdefault(
        "mlflow_tracking_uri",
        os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI),
    )
    config.setdefault("backend", {"module": "backend.Local"})
    return config


def load_project_configuration(path: str | Path = DEFAULT_PROJECT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return parse_project_configuration(None)
    return parse_project_configuration(config_path.read_text())


def configure_logger(project: dict[str, Any] | None = None) -> logging.Logger:
    project = project or {}
    logging_file = project.get("logging", "logging.conf")

    if Path(str(logging_file)).exists():
        logging.config.fileConfig(str(logging_file))
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )

    return logging.getLogger("nyc-taxi")


def configure_mlflow_tracking(mlflow_tracking_uri: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(mlflow_tracking_uri)


def _optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    for column in df.columns:
        column_type = df[column].dtype
        if column_type == "int64":
            column_min = df[column].min()
            column_max = df[column].max()
            if column_min > np.iinfo(np.int32).min and column_max < np.iinfo(np.int32).max:
                df[column] = df[column].astype(np.int32)
            elif column_min > np.iinfo(np.int16).min and column_max < np.iinfo(np.int16).max:
                df[column] = df[column].astype(np.int16)
        elif column_type == "float64":
            df[column] = pd.to_numeric(df[column], downcast="float")

    return df


def _load_parquet_in_chunks(
    file_path: str | Path,
    logger: logging.Logger,
    sample_size: int | None = None,
    use_row_groups: bool = True,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.num_row_groups

    logger.info(f"Total rows in file: {total_rows:,}")
    logger.info(f"Number of row groups: {num_row_groups}")

    if sample_size and sample_size < total_rows:
        logger.info(f"Sampling {sample_size:,} rows...")
        if use_row_groups and num_row_groups > 1:
            sample_per_group = max(1, sample_size // num_row_groups)
            chunks = []
            for index in range(num_row_groups):
                chunk = parquet_file.read_row_group(index).to_pandas()
                if len(chunk) > sample_per_group:
                    chunk = chunk.sample(n=min(sample_per_group, len(chunk)), random_state=42)
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            if len(df) > sample_size:
                return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            return df

        df = pd.read_parquet(file_path, engine="pyarrow")
        return df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)

    logger.info(f"Loading full dataset in {num_row_groups} row groups...")
    chunks = [parquet_file.read_row_group(index).to_pandas() for index in range(num_row_groups)]
    return pd.concat(chunks, ignore_index=True)


def load_dataset(
    dataset_path: str | Path,
    logger: logging.Logger,
    run_id: str,
    output_dir: str | Path = "processed_dataset",
    production: bool = False,
) -> str | None:
    """
    Load, shuffle, and stage a dataset as parquet for downstream training steps.
    """
    import gc

    path = Path(dataset_path)
    if not path.exists():
        return None

    if path.suffix == ".parquet":
        data = _load_parquet_in_chunks(path, logger)
        data = _optimize_dataframe_dtypes(data)
    else:
        data = pd.read_csv(path)

    seed = int(time.time() * 1000) if production else 47
    data = data.sample(frac=1, random_state=seed)
    logger.info(f"Loaded dataset with {len(data)} samples")

    output_path = Path(output_dir) / f"processed_dataset_{run_id}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path)

    del data
    gc.collect()
    logger.info(f"Saved staged dataset to {output_path}")
    return str(output_path)


@dataclass
class Pipeline:
    """Plain base object shared by local training and inference runners."""

    project_config_path: str = DEFAULT_PROJECT_CONFIG
    dataset: str = DEFAULT_DATASET
    mlflow_tracking_uri: str | None = None
    run_id: str = field(default_factory=generate_run_id)
    production: bool = False
    project: dict[str, Any] = field(init=False)
    logger: logging.Logger = field(init=False)
    mode: str = field(init=False)

    def __post_init__(self) -> None:
        self.project = load_project_configuration(self.project_config_path)
        self.mlflow_tracking_uri = (
            self.mlflow_tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or str(self.project.get("mlflow_tracking_uri", DEFAULT_MLFLOW_TRACKING_URI))
        )
        self.logger = configure_logger(self.project)
        self.mode = "production" if self.production else "development"
        configure_mlflow_tracking(self.mlflow_tracking_uri)

    def prepare_dataset(self) -> str:
        data_path = load_dataset(
            self.dataset,
            logger=self.logger,
            run_id=self.run_id,
            production=self.production,
        )
        if data_path is None:
            raise FileNotFoundError(f"Dataset file not found: {self.dataset}")
        return data_path
