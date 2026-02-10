# 🚕 NYC Taxi Fare Prediction

**Production-grade ML system for NYC taxi fare prediction. ONNX Runtime inference, self-hosted MLflow, Docker Compose infrastructure, Kubeflow Pipelines orchestration, and a unified feature engineering module that eliminates training-serving skew.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Metaflow](https://img.shields.io/badge/pipeline-Metaflow-purple.svg)](https://metaflow.org/)
[![KFP v2](https://img.shields.io/badge/pipeline-Kubeflow_Pipelines-blue.svg)](https://www.kubeflow.org/docs/components/pipelines/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blue.svg)](https://mlflow.org/)
[![ONNX](https://img.shields.io/badge/inference-ONNX_Runtime-orange.svg)](https://onnxruntime.ai/)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/infra-Docker-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## System Architecture

![SYSTEM ARCHITECTURE](extras/System-arch.png)

---

## Quick Start

### Option A: Docker Compose (Recommended)

```bash
# 1. Clone and configure
git clone https://github.com/nilesh-auradkar05/NYC-Taxi-Fare-Prediction-ML-System.git
cd NYC-Taxi-Fare-Prediction-ML-System
cp .env.example .env

# 2. Start infrastructure (MLflow + MinIO)
docker-compose up mlflow minio minio-setup -d

# 3. Train a model (registers ONNX model to MLflow)
python src/pipelines/training.py run

# 4. Download model artifacts to local cache
python src/serving/download_model.py

# 5. Start the full stack (adds API service)
docker-compose up -d

# 6. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pickup_datetime": "2025-01-15 08:30:00",
       "dropoff_datetime": "2025-01-15 09:15:00",
       "trip_distance": 5.2,
       "passenger_count": 2}'
```

**Services:**

| Service | URL | Purpose |
|---------|-----|---------|
| **API** | http://localhost:8000 | Fare predictions (Swagger at `/docs`) |
| **MLflow** | http://localhost:5000 | Experiment tracking UI |
| **MinIO** | http://localhost:9001 | S3 artifact storage console |

### Option B: Local (Without Docker)

```bash
# Install dependencies
poetry install  # or: pip install -r requirements.txt

# Train
python src/pipelines/training.py run

# Serve
python src/serving/download_model.py
uvicorn src.serving.api:app --port 8000
```

### Option C: Kubeflow Pipelines

```bash
# Build the training image
docker build -f Dockerfile.training -t nyc-taxi-training .

# Compile pipeline to YAML
python src/pipelines/kfp_pipeline.py --output pipeline.yaml

# Submit to a running Kubeflow cluster
python src/pipelines/kfp_pipeline.py --submit --host http://kubeflow.local
```

**Launch the UI:**
```bash
streamlit run src/serving/frontend.py
```

---

## What's Inside

This isn't a Jupyter notebook that "kind of works." It's a complete MLOps system with:

| Component | What it does | Tech |
|-----------|--------------|------|
| **Training Pipeline** | Loads data → engineers 29 features → trains XGBoost → exports ONNX → registers model | Metaflow / KFP v2 |
| **Inference Pipeline** | Batch predictions on new data via ONNX Runtime | Metaflow |
| **Model Registry** | Versioned ONNX models with experiment lineage | Self-hosted MLflow |
| **Experiment Tracking** | Metrics, parameters, artifacts logged automatically | MLflow |
| **Feature Engineering** | Single source of truth — same function in training, inference, and serving | `common/features.py` |
| **REST API** | Sub-100ms ONNX Runtime predictions with health checks | FastAPI |
| **Infrastructure** | MLflow + MinIO + API in one command | Docker Compose |
| **Web UI** | User-friendly fare estimator with tip calculator | Streamlit |

---

## Pipeline Details

### Training Pipeline

![Training Pipeline](extras/Training%20Pipeline.png)

The training pipeline runs in two flavors that share the same `common/features.py` module and produce the same ONNX model format:

**Metaflow** — `python src/pipelines/training.py run` — single-process execution with `foreach` parallelism for cross-validation. Good for local development and iteration.

**Kubeflow Pipelines** — `python src/pipelines/kfp_pipeline.py` — each step runs in its own container via `dsl.ParallelFor`. Registration is gated by `dsl.Condition` on the R² threshold. Production-ready on Kubernetes.

### Inference Pipeline

![Inference Pipeline](extras/Inference%20Pipeline.png)

---

## Serving Architecture

![Serving Architecture](extras/Serving-arch.png)

The serving layer loads two artifacts: `model.onnx` (ONNX Runtime) and `transformer.joblib` (sklearn ColumnTransformer). Feature engineering uses the same `engineer_features()` function from `common/features.py` — no duplication, no branching, no training-serving skew.

---

## Project Structure

```
.
├── Dockerfile                    # Multi-stage serving image (FastAPI + ONNX Runtime)
├── Dockerfile.training           # Base image for KFP pipeline components
├── docker-compose.yml            # Full stack: MLflow + MinIO + API
├── docker-compose.mlflow.yml     # MLflow + MinIO only
├── requirements-serving.txt      # Minimal API dependencies
├── requirements-training.txt     # Training + KFP dependencies
├── .env.example                  # Environment variable template
├── .dockerignore
│
├── src/
│   ├── common/
│   │   ├── features.py           # Single source of truth for feature engineering
│   │   └── pipeline.py           # Base Metaflow pipeline class + decorators
│   │
│   ├── pipelines/
│   │   ├── kfp_components.py     # KFP v2 container components (6 steps)
│   │   ├── kfp_pipeline.py       # KFP pipeline DAG + compiler/submitter
│   │   ├── training.py           # Metaflow training pipeline
│   │   └── inference.py          # Metaflow batch inference (ONNX Runtime)
│   │
│   └── serving/
│       ├── api.py                # FastAPI prediction service (ONNX Runtime)
│       ├── download_model.py     # Pull model from MLflow to local cache
│       └── frontend.py           # Streamlit UI
│
├── dbt_nyc_taxi/                 # dbt models (Snowflake feature transforms)
│   ├── models/
│   │   ├── staging/              # Raw data cleaning
│   │   ├── intermediate/         # Business logic
│   │   └── features/             # ML-ready feature tables
│   └── dbt_project.yml
│
├── config/
│   └── local.yml                 # Environment configuration
├── Dataset/                      # Parquet files (DVC-tracked, not in repo)
├── models/cache/                 # Locally cached model artifacts
└── predictions/                  # Inference outputs
```

---

## Feature Engineering

The model learns from **29 engineered features** extracted from raw trip data:

![Feature Engineering Mindmap](extras/Feature-eng-mindmap.png)

All feature logic lives in a single function — `engineer_features()` in `src/common/features.py`. Training, inference, and the API all import this same function with no arguments that change behavior. This eliminates the most common source of silent prediction drift in production ML systems.

**Data Leakage Fix:** Three features in the original pipeline used `total_amount` (the prediction target) during computation, inflating metrics and causing training-serving skew:

| Feature | Before (Leaking) | After (Fixed) |
|---------|-------------------|---------------|
| `revenue_per_mile` | `total_amount / trip_distance` | **Removed** — redundant with `fare_per_mile` |
| `tip_percentage` | `tip_amount / total_amount` | `tip_amount / fare_amount` |
| `has_negative_fare` | `total_amount < 0` | `fare_amount < 0` |

---

## Model Performance

Trained on **4.2 million** NYC yellow taxi trips (September 2025):

| Metric | Value |
|--------|-------|
| **R² Score** | 0.87 |
| **RMSE** | $4.23 |
| **MAE** | $2.89 |
| **Training Time** | ~8 minutes |

> **Note:** These metrics were recorded before the data leakage fix. Post-fix R² is expected to be lower — that drop reflects honest evaluation. The previous metrics were inflated by features derived from the target variable.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check for load balancers |
| `/model/info` | GET | Model version, type, and metadata |
| `/predict` | POST | Single trip fare prediction |

**Example Response:**
```json
{
  "predicted_fare": 25.50,
  "trip_duration_minutes": 45.0,
  "model_version": "1",
  "prediction_timestamp": "2025-01-15T10:30:00"
}
```

---

## Tech Stack

![Tech Stack](extras/tech-stack.png)

| Layer | Tool | Purpose |
|-------|------|---------|
| **Training** | XGBoost, scikit-learn | Model training with cross-validation |
| **Orchestration** | Metaflow, Kubeflow Pipelines v2 | Pipeline execution (local and K8s) |
| **Model Format** | ONNX | Framework-agnostic, 2-5x faster inference |
| **Experiment Tracking** | MLflow (self-hosted) | Metrics, parameters, model registry |
| **Artifact Storage** | MinIO / AWS S3 | ONNX models, transformers, datasets |
| **Serving** | FastAPI, ONNX Runtime | Real-time predictions |
| **Containerization** | Docker, Docker Compose | Reproducible environments |
| **Data Versioning** | DVC + S3 | Git-like versioning for datasets |
| **Feature Transforms** | dbt + Snowflake | SQL-based feature pipelines |
| **Frontend** | Streamlit | Development UI |

---

## Key Design Decisions

### Why ONNX Runtime?

The training pipeline trains XGBoost normally, then converts to ONNX format before registering with MLflow. The serving layer loads the `.onnx` file directly into ONNX Runtime — no XGBoost needed at inference time. This gives you 2-5x faster predictions, a framework-agnostic model format, and a smaller serving image.

### Why Self-Hosted MLflow?

MLflow runs on your infrastructure with S3-compatible storage (MinIO for local dev, real S3 in production). No Databricks authentication required in serving containers. Full experiment tracking and model versioning without vendor lock-in.

### Why Two Orchestrators?

Metaflow and KFP v2 coexist — same `features.py`, same ONNX format, same MLflow registry. Metaflow is simpler for local development. KFP runs on Kubernetes with true container isolation per step, parallel cross-validation via `dsl.ParallelFor`, and conditional model registration via `dsl.Condition`.

### Why a Unified Feature Module?

Feature duplication across training and serving pipelines is the #1 cause of silent prediction degradation. `src/common/features.py` is the single source of truth — one function, no parameters that change behavior, no branching. Change a feature here and it updates everywhere.

### Why Multi-Stage Docker?

The serving image contains only what the API needs: FastAPI, ONNX Runtime, sklearn (for the transformer), pandas, numpy. No XGBoost, no training dependencies, no compiler tools. Builder stage handles compilation, runtime stage stays slim. Non-root user, built-in health checks, ready for K8s probes.

---

## Configuration

Set these environment variables (or use `.env.example` as a template):

```bash
# MLflow (self-hosted)
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME=nyc-taxi-experiment

# S3 / MinIO (for MLflow artifact storage)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000   # MinIO only; omit for real S3

# Model cache (for serving)
MODEL_CACHE_DIR=models/cache
```

---

## Installation

```bash
# Clone
git clone https://github.com/nilesh-auradkar05/NYC-Taxi-Fare-Prediction-ML-System.git
cd NYC-Taxi-Fare-Prediction-ML-System

# Install dependencies
poetry install

# Or with pip
pip install -r requirements.txt
```

**Core Dependencies:**
- Python 3.11+
- XGBoost, scikit-learn, pandas, numpy
- ONNX Runtime, onnxmltools
- Metaflow, MLflow
- FastAPI, uvicorn, streamlit
- kfp (for Kubeflow Pipelines)
- loguru, python-dotenv

---

## Data

Download NYC TLC trip data from the [official source](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

```bash
# Example: September 2025 Yellow Taxi data
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-09.parquet \
  -O Dataset/yellow_tripdata_2025-09.parquet
```

---

## Roadmap

![Project Milestone](extras/project-milestone.png)

- [x] Training pipeline with Metaflow
- [x] MLflow experiment tracking
- [x] FastAPI serving layer
- [x] Streamlit frontend
- [x] Batch inference pipeline
- [x] Unified feature engineering (data leakage fix)
- [x] ONNX model export + ONNX Runtime inference
- [x] Self-hosted MLflow (replaced Databricks/Unity Catalog)
- [x] Docker containerization (multi-stage builds)
- [x] Kubeflow Pipelines v2 (container components)
- [x] Data versioning (DVC + S3)
- [x] dbt + Snowflake feature transforms
- [ ] Model monitoring & drift detection (Evidently, Prometheus, Grafana)
- [ ] React/Next.js frontend
- [ ] Cloud deployment (AWS EKS)

---

## Why This Project?

**A reference architecture for production-grade ML systems, demonstrating how the pieces of a real MLOps platform fit together:**

| Question | Answer |
|----------|--------|
| How do you version models? | Self-hosted MLflow Model Registry with ONNX artifacts in S3 |
| How do you serve predictions? | FastAPI + ONNX Runtime with sub-100ms latency |
| How do you track experiments? | MLflow logging every run automatically |
| How do you handle training-serving skew? | Single `engineer_features()` function — same code everywhere |
| How do you orchestrate? | Metaflow (local dev) + Kubeflow Pipelines (Kubernetes) |
| How do you deploy? | Docker Compose (local) → AWS EKS (production) |

---

## Acknowledgments

- [ml.school](https://www.ml.school) curriculum for the project structure
- [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) for the trip data
- [MLflow](https://mlflow.org/) for experiment tracking and model registry
- [Kubeflow](https://www.kubeflow.org/) for pipeline orchestration on Kubernetes

---