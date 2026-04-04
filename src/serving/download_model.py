"""
Model Download Script for NYC Taxi Fare Prediction

This script downloads ONNX model and preprocessing transformer from Mlflow registry and caches them locally
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
import tempfile
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def validate_env():
    """
    Validate required environment variables are set(MLFLOW_TRACKING_URI).
    
    Raises:
        EnvironmentError: If required variables are not set
    """
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')

    if not tracking_uri:
        logger.error("MLFlow tracking URI is not set.")
        logger.error("Set it in your .env file or environment variables")
        logger.error("\tMLFLOW_TRACKING_URI=http://127.0.0.1:5000")
        raise EnvironmentError("Missing MLFLOW_TRACKING_URI")

    logger.info(f"Mlflow tracking URI: {tracking_uri}")

def download_model(
    model_name: str="nyc-taxi-model",
    model_version: str="latest",
    output_dir: str="models/cache",
) -> dict:
    """
    Download ONNX model and transformer from MLflow to local cache.

    Parameters:
    -----------
    model_name : str
        Default: "nyc-taxi-model"
    
    model_version : str
        Default: "latest"
    
    output_dir : str
        Local directory to cache the model artifacts.
        Default: "models/cache"
    
    Returns:
    --------
    dict
        Dictionary containing paths to downloaded artifacts:
        {
            "model_path": "/path/to/model.onnx",
            "transformer_path": "/path/to/transformer.joblib",
            "metadata_path": metadata.json
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

    logger.info("="*60)
    logger.info("ONNX Model download")
    logger.info("="*60)

    logger.info("Configuring MLflow....")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 'http://127.0.0.1:5000')
    mlflow.set_tracking_uri(tracking_uri)

    # Create Output Directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")

    # Resolve model version
    client = MlflowClient()

    logger.info(f"\nModel: {model_name}")
    logger.info(f"Requested version: {model_version}")

    try:
        if model_version == "latest":
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise RuntimeError(f"No versions found for '{model_name}'")

            version_info = sorted(versions, key=lambda v: int(v.version))[-1]
        else:
            version_info = client.get_model_version(model_name, model_version)

        version_number = version_info.version
        run_id = version_info.run_id
        logger.info(f"\tResolved: version {version_number}, run {run_id}")

    except Exception as e:
        logger.error(f"Failed to resolve model version: {e}")
        raise RuntimeError(f"Could not find model {model_name} version {model_version}") from e

    # Download Model

    logger.info("\n Downloading ONNX model....")
    try:
        with tempfile.TemporaryDirectory() as  tmp_dir:
            artifact_dir = client.download_artifacts(str(run_id), "onnx_model", tmp_dir)
            source_onnx = Path(artifact_dir) / "model.onnx"

            if not source_onnx.exists():
                found = list(Path(artifact_dir).rglob("*.onnx"))
                if found:
                    source_onnx = found[0]
                else:
                    raise FileNotFoundError(
                        "No .onnx file found in artifacts. "
                        f"Contents: {list(Path(artifact_dir).rglob("*.onnx"))}"
                    )

            model_path = output_path / "model.onnx"
            import shutil
            shutil.copy2(source_onnx, model_path)
            logger.info(f"\tSaved: {model_path} ({model_path.stat().st_size/1024:.0f} KB)")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(f"Model download failed: {e}") from e

    # Download Transformer
    logger.info(f"\nDownloading preprocessing transformer from run: {run_id}")
    try:
        with tempfile.TemporaryDirectory() as  tmp_dir:
            artifact_dir = client.download_artifacts(str(run_id), "preprocessing", str(tmp_dir))
            source_transformer = Path(artifact_dir) / "features.joblib"

            if not source_transformer.exists():
                raise FileNotFoundError(
                    f"Transformer not found at: {source_transformer}"
                )

            transformer_path = output_path / "transformer.joblib"
            shutil.copy2(source_transformer, transformer_path)

            logger.info("    Transformer downloaded successfully and cached")
            logger.info(f"    Type: {type(transformer_path).__name__}")
            logger.info(f"    Saved at: {transformer_path}")

    except Exception as e:
        logger.error(f"Failed to download transformer: {e}")
        raise RuntimeError(f"Transformer download failed: {e}") from e

    import json

    # Save Metadata for the serving layer
    metadata = {
        "model_name": model_name,
        "model_version": str(version_number),
        "model_alias": None,
        "model_format": "ONNX",
        "run_id": run_id,
        "mlflow_tracking_uri": tracking_uri,
        "model_path": str(model_path),
        "transformer_path": str(transformer_path),
    }

    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
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
        description="Download NYC Taxi model from MLflow registry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-name",
        default="nyc-taxi-model",
        help="Registered model name in MLflow (default: nyc-taxi-model)"
    )

    parser.add_argument(
        "--model-version",
        default="latest",
        help="Model version: 'latest' for model name in MLflow (default: latest)"
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
        download_model(
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