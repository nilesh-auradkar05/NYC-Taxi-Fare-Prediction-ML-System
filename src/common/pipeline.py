"""
This file contains the common pipeline decorators and base Pipeline class.
"""

import importlib
import os
import re
import sys
import time
from contextlib import suppress
from pathlib import Path

import pandas as pd
import yaml
from metaflow import (
    Config,
    FlowMutator,
    FlowSpec,
    Parameter,
    config_expr,
    current,
    project,
    user_step_decorator,
)


@user_step_decorator
def dataset(step_name, flow, inputs=None, attr=None):
    """
    Load and prepare dataset.

    Refactored to save the processed dataframe to a local Parquet file
    and pass the file path, avoiding Metaflow's pickling bottleneck.
    """
    import gc
    import os
    import numpy as np
    import pyarrow.parquet as pq

    # Check if dataset file exists
    if not Path(flow.dataset).exists():
        flow.data_path = None
        yield
    else:
        if str(flow.dataset).endswith(".parquet"):
            # Use memory-efficient chunked loading for parquet
            def load_parquet_in_chunks(
                file_path, sample_size=None, use_row_groups=True
            ):
                """
                Load parquet file in chunks to save memory.
                """
                parquet_file = pq.ParquetFile(file_path)
                total_rows = parquet_file.metadata.num_rows
                num_row_groups = parquet_file.num_row_groups

                flow.logger.info(f"Total rows in file: {total_rows:,}")
                flow.logger.info(f"Number of row groups: {num_row_groups}")

                if sample_size and sample_size < total_rows:
                    flow.logger.info(f"Sampling {sample_size:,} rows...")
                    if use_row_groups and num_row_groups > 1:
                        sample_per_group = max(1, sample_size // num_row_groups)
                        chunks = []
                        for i in range(num_row_groups):
                            chunk = parquet_file.read_row_group(i).to_pandas()
                            if len(chunk) > sample_per_group:
                                chunk = chunk.sample(
                                    n=min(sample_per_group, len(chunk)), random_state=42
                                )
                            chunks.append(chunk)
                        df = pd.concat(chunks, ignore_index=True)
                        if len(df) > sample_size:
                            df = df.sample(n=sample_size, random_state=42).reset_index(
                                drop=True
                            )
                    else:
                        df = pd.read_parquet(file_path, engine="pyarrow")
                        df = df.sample(
                            n=min(sample_size, len(df)), random_state=42
                        ).reset_index(drop=True)
                else:
                    flow.logger.info(
                        f"Loading full dataset in {num_row_groups} row groups..."
                    )
                    chunks = []
                    for i in range(num_row_groups):
                        chunk = parquet_file.read_row_group(i).to_pandas()
                        chunks.append(chunk)

                    df = pd.concat(chunks, ignore_index=True)

                # Optimize dtypes to save memory
                flow.logger.info("Optimizing data types to save memory...")
                for col in df.columns:
                    col_type = df[col].dtype
                    if col_type == "int64":
                        c_min = df[col].min()
                        c_max = df[col].max()
                        if (
                            c_min > np.iinfo(np.int32).min
                            and c_max < np.iinfo(np.int32).max
                        ):
                            df[col] = df[col].astype(np.int32)
                        elif (
                            c_min > np.iinfo(np.int16).min
                            and c_max < np.iinfo(np.int16).max
                        ):
                            df[col] = df[col].astype(np.int16)
                    elif col_type == "float64":
                        df[col] = pd.to_numeric(df[col], downcast="float")

                flow.logger.info(f"Loaded {len(df):,} rows total")
                return df

            # Load the data
            data = load_parquet_in_chunks(flow.dataset, sample_size=None)

        else:
            data = pd.read_csv(flow.dataset)

        # Shuffle Logic
        seed = int(time.time() * 1000) if current.is_production else 47
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        flow.logger.info(f"Loaded dataset with {len(data)} samples")

        # --- CHANGED: Save to disk instead of attaching to flow.data ---

        # Generate a unique filename using run_id to avoid overwriting if running concurrently
        filename = f"processed_dataset_{current.run_id}.parquet"
        output_path = Path(os.getcwd()) / "processed_dataset" / filename

        flow.logger.info(f"Saving processed dataframe to {output_path}...")

        # Save to parquet
        data.to_parquet(output_path)

        # Pass the PATH as the artifact, not the dataframe
        flow.data_path = str(output_path)

        # Explicit garbage collection to free RAM immediately
        del data
        gc.collect()
        flow.logger.info("Memory cleaned. Proceeding to next step.")

        yield


@user_step_decorator
def logging(step_name, flow, inputs=None, attr=None):
    """
    Configure the logging handler.

    This decorator configures the logging handler to log on every individual step of a pipeline.
    This decorator will do that, and set an artifact so every step in the flow can access it.
    """
    import logging
    import logging.config

    # Fetching the logging configuration file from the project settings.
    logging_file = flow.project.get("logging", "logging.conf")

    if Path(logging_file).exists():
        logging.config.fileConfig(logging_file)
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )

    flow.logger = logging.getLogger("nyc-taxi")
    yield


@user_step_decorator
def mlflow(step_name, flow, inputs=None, attr=None):
    """Configure MLFlow's tracking URI for the current step."""
    import mlflow

    mlflow.set_tracking_uri(flow.mlflow_tracking_uri)
    yield


def parse_project_configuration(x):
    """
    Parse the project configuration from the supplied files.

    This function will expand any environment variables that are present in the
    backend configuration values. The environment variables should be in the format
    `${ENVIRONMENT_VARIABLE}`.
    """
    config = yaml.full_load(x)

    # If the mlflow tracking uri is not part of the configuration, we will set it to the
    # `MLFLOW_TRACKING_URI` environment variable.
    if "mlflow_tracking_uri" not in config:
        config["mlflow_tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI", "databricks")

    if "backend" not in config:
        config["backend"] = {"module": "backend.Local"}

    # This regex pattern matches any environment variable in the format ${ENVIRONMENT_VARIABLE}
    pattern = re.compile(r"\$\{(\w+)\}")

    def replacer(match):
        env_var = match.group(1)
        return os.getenv(env_var, f"${{{env_var}}}")

    for key, value in config.items():
        if isinstance(value, str):
            config["backend"][key] = pattern.sub(replacer, value)

    return config


class pipeline(FlowMutator):
    """Mutate a flow by applying a set of decorators to every step."""

    def mutate(self, mutable_flow):
        """Mutates the supplied flow."""
        for _, step in mutable_flow.steps:
            # Letting every step to have access to a pre-configured logger.
            step.add_decorator("logging", duplicates=step.IGNORE)

            # Letting every step configure the Mlflow tracking URI.
            step.add_decorator("mlflow", duplicates=step.IGNORE)


@pipeline
@project(name=config_expr("project.project"))
class Pipeline(FlowSpec):
    """Foundation flow for pipelines that require access to the dataset and backend"""

    project = Config(
        "project",
        help="Project Configuration settings.",
        default="config/local.yml",
        parser=parse_project_configuration,
    )

    dataset = Parameter(
        "dataset",
        help="Project dataset that will be used to train and evaluate the model.",
        default="Dataset/yellow_tripdata_2025-09.parquet",
    )

    mlflow_tracking_uri = Parameter(
        "mlflow-tracking-uri",
        help="MLflow tracking URI.",
        default="127.0.0.1:5000",
    )
