"""
Model Download Script for NYC Taxi Fare Prediction

This script downloads the model and preprocessing transformer from Databricks Unity Catalog and caches them locally
for use by the FastAPI serving layer.

Usage:
-------------
    # Download the champion model (default)
    python or python3 src/serving/download_model.py

    # Download a specific model version
    python or python3 src/serving/download_model.py --model-version 2

    # Download to custom directory
    python or python3 src/serving/download_model.py --output-dir /path/to/cache

"""

import os
import sys
import argparse
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def validate_env():
    """
    Validate required environment variables are set.

    Databricks Unity Catalog requires authentication via:
    - DATABRICKS_HOST: The workspace URL (e.g., https://xxx.cloud.databricks.com)
    - DATABRICKS_TOKEN: Personal Access Token (PAT) for API access
    
    These are typically stored in a .env file for local development
    and injected as secrets in production environments.
    
    Raises:
        EnvironmentError: If required variables are not set
    """
    required_vars = ["DATABRICKS_HOST", "DATABRICKS_TOKEN"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Please set these in your .env file or environment")
        logger.error("")
        logger.error("DATABRICKS_HOST=<your_databricks_host>")
        logger.error("DATABRICKS_TOKEN=<your_databricks_token>")
        raise EnvironmentError(f"Missing required environment variables: {missing}")

    logger.info("Environment variables validated successfully")
    logger.info(f"DATABRICKS_HOST: {os.getenv("DATABRICKS_HOST")}")

def download_model(
    model_name: str="ml_models.nyc-taxi.nyc-taxi-model",
    model_version: str="champion",
    output_dir: str="models/cache",
) -> dict:
    """
    Download model and transformer from unity catalog to local cache.

    This function connects to databricks unity catalog, downloads the
    specified model version or alias, and saves it locally along with
    the preprocessing transformer for faster inference and to avoid
    redundant model download.

    Parameters:
    -----------
    model_name : str
        Full Unity Catalog model path: <catalog>.<schema>.<model_name>
        Default: "ml_models.nyc-taxi.nyc-taxi-model"
    
    model_version : str
        Version or alias to download:
        - "champion": The production-ready model (recommended)
        - "challenger": Model being tested
        - "<number>": Specific version number (e.g., "1", "2")
        Default: "champion"
    
    output_dir : str
        Local directory to cache the model artifacts.
        Default: "models/cache"
    
    Returns:
    --------
    dict
        Dictionary containing paths to downloaded artifacts:
        {
            "model_path": "/path/to/model.joblib",
            "transformer_path": "/path/to/transformer.joblib",
            "model_version": "1",
            "model_name": "ml_models.nyc-taxi.nyc-taxi-model"
        }
    
    Raises:
    -------
    RuntimeError
        If model or transformer download fails
    
    Example:
    --------
        >>> paths = download_model()
        >>> print(paths["model_path"])
        "models/cache/model.joblib"
    """

    import mlflow
    from mlflow.tracking import MlflowClient
    import joblib
    import tempfile

    logger.info("="*60)
    logger.info("Model download from unity catalog")
    logger.info("="*60)

    # Configure MLflow for databricks
    # Set both tracking URI and registry URI to databricks
    # - tracking_uri="databricks"
    # - registry_uri="databricks-uc"

    logger.info("Configuring MLflow for databricks....")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    logger.info("    Tracking URI: databricks")
    logger.info("    Registry URI: databricks-uc")

    # Create Output Directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")

    # Resolve model version
    client = MlflowClient()

    logger.info(f"\nModel: {model_name}")
    logger.info(f"Requested version: {model_version}")

    try:
        # Check if this is an alias or a version number
        is_alias = model_version in ["champion", "challenger"] or model_version.startswith("@")
        alias = None

        if is_alias:
            # Resolve alias to version number
            alias = model_version.lstrip("@")
            version_info = client.get_model_version_by_alias(
                name=model_name,
                alias=alias,
            )
            version_number = version_info.version
            run_id = version_info.run_id
            logger.info(f"Resolved @{alias} to version {version_number}")
        else:
            # Use specific version number
            version_number = model_version
            version_info = client.get_model_version(model_name, version_number)
            run_id = version_info.run_id

        logger.info(f"    Version: {version_number}")
        logger.info(f"    Run ID: {run_id}")

    except Exception as e:
        logger.error(f"Failed to resolve model version: {e}")
        raise RuntimeError(f"Could not find model {model_name} version {model_version}") from e

    # Download Model

    logger.info(f"\n Downloading model from unity catalog....")
    try:
        if is_alias and alias is not None:
            model_uri = f"models:/{model_name}@{alias}"
        else:
            model_uri = f"models:/{model_name}/{version_number}"

        logger.info(f"    Model URI: {model_uri}")

        # Download and load the model
        model = mlflow.sklearn.load_model(model_uri)

        # Save locally using joblib
        model_path = output_path / "model.joblib"
        joblib.dump(model, model_path)

        logger.info(f"    Model downloaded successfully and cached")
        logger.info(f"    Type: {type(model).__name__}")
        logger.info(f"    Saved at: {model_path}")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(f"Model download failed: {e}") from e

    # Download Transformer
    logger.info(f"\nDownloading preprocessing transformer from run: {run_id}")
    try:
        with tempfile.TemporaryDirectory() as  tmp_dir:
            # Download preprocessing artifacts from the training run
            artifact_path = client.download_artifacts(
                run_id,
                "preprocessing",
                tmp_dir
            )

            # Load the transformer
            source_transformer_path = Path(artifact_path) / "features.joblib"

            if not source_transformer_path.exists():
                raise FileNotFoundError(
                    f"Transformer not found at: {source_transformer_path}"
                )

            transformer = joblib.load(source_transformer_path)

            # Save locally using joblib
            transformer_path = output_path / "transformer.joblib"
            joblib.dump(transformer, transformer_path)

            logger.info("    Transformer downloaded successfully and cached")
            logger.info(f"    Type: {type(transformer).__name__}")
            logger.info(f"    Saved at: {transformer_path}")

    except Exception as e:
        logger.error(f"Failed to download transformer: {e}")
        raise RuntimeError(f"Transformer download failed: {e}") from e

    # Save metadata
    import json

    metadata = {
        "model_name": model_name,
        "model_version": str(version_number),
        "model_alias": model_version if is_alias and alias is not None else None,
        "run_id": run_id,
        "download_timestamp": str(Path(model_path).stat().st_mtime),
        "model_path": str(model_path),
        "transformer_path": str(transformer_path),
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"    Metadata saved at: {metadata_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model:       {model_path}")
    logger.info(f"Transformer: {transformer_path}")
    logger.info(f"Metadata:    {metadata_path}")
    logger.info("=" * 60)

    return {
        "model_path": str(model_path),
        "transformer_path": str(transformer_path),
        "metadata_path": str(metadata_path),
        "model_version": str(version_number),
        "model_name": model_name,
    }

def main():
    """
    Main entry point for the model download script

    Parses command line arguments and downloads the model and transformer
    from unity catalog to local cache
    """
    parser = argparse.ArgumentParser(
        description="Download NYC Taxi model from unity catalog",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
        Examples:
        # Download champion model (default)
        python download_model.py
        
        # Download specific version
        python download_model.py --model-version 2
        
        # Download challenger model
        python download_model.py --model-version challenger
        
        # Custom output directory
        python download_model.py --output-dir /tmp/model_cache
        """
    )

    parser.add_argument(
        "--model-name",
        default="ml_models.nyc-taxi.nyc-taxi-model",
        help="Full unity catalog model path (default: ml_models.nyc-taxi.nyc-taxi-model)"
    )

    parser.add_argument(
        "--model-version",
        default="champion",
        help="Model version or alias: 'champion', 'challenger', or version number (default: champion)"
    )

    parser.add_argument(
        "--output-dir",
        default="models/cache",
        help="Local directory to cache model artifacts (default: models/cache)"
    )

    args = parser.parse_args()

    # Execute Download
    try:
        # Validate environment variables
        validate_env()

        # Download model and transformer
        paths = download_model(
            model_name=args.model_name,
            model_version=args.model_version,
            output_dir=args.output_dir,
        )

        logger.info("\n Model ready for serving!")
        logger.info("    Start the API with: uvicorn src.serving.api:app --reload")

    except Exception as e:
        logger.error(f"\n Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()