import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


class Settings:
    APP_NAME = os.getenv("APP_NAME", "arabic_sentiment_analysis")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "on"}

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    MODEL_API_PORT = int(os.getenv("MODEL_API_PORT", "8000"))
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
    GRAFANA_PORT = int(os.getenv("GRAFANA_PORT", "3000"))

    MODEL_TOKENIZER = os.getenv("MODEL_TOKENIZER", "aubmindlab/bert-base-arabertv02")
    MODEL_VOCAB_PATH = Path(
        os.getenv("MODEL_VOCAB_PATH", str(BASE_DIR / "notebooks" / "vocab.pkl"))
    )
    MODEL_ONNX_PATH = Path(
        os.getenv("MODEL_ONNX_PATH", str(BASE_DIR / "notebooks" / "model.onnx"))
    )
    BASELINE_PATH = Path(
        os.getenv("BASELINE_PATH", str(BASE_DIR / "data" / "training_baseline.json"))
    )
    PREDICTIONS_DIR = Path(
        os.getenv("PREDICTIONS_DIR", str(BASE_DIR / "data" / "scoring" / "output"))
    )
    REPORT_PATH = Path(
        os.getenv("REPORT_PATH", str(BASE_DIR / "reports" / "drift_report.html"))
    )

    GAUGES_URL = os.getenv("GAUGES_URL", "http://model-api:8000/internal/gauges")
    PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
    GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana:3000")
    GRAFANA_ADMIN_USER = os.getenv("GRAFANA_ADMIN_USER", "admin")
    GRAFANA_ADMIN_PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD", "admin")

    BENTOML_SERVICE_NAME = os.getenv("BENTOML_SERVICE_NAME", "arabic_sentiment_service")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000/")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")


settings = Settings()
