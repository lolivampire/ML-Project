# services/model_service.py
# File ini murni business logic dan "buta" terhadap FastAPI.

import numpy as np
import logging
from typing import Any
from app.config import settings
from app.repositories.prediction_repo import PredictionRepository

logger = logging.getLogger(__name__)

RISK_THRESHOLDS = {
    "low":    (0.0, 0.40),
    "medium": (0.40, 0.70),
    "high":   (0.70, 1.0),
}

class ModelService:
    def __init__(self, model: Any, repo: PredictionRepository, version: str = "1.0.0"):
        self.model = model
        self.repo = repo
        self.version = version

    def predict(self, features: list[float]) -> dict:
        if not features:
            raise ValueError("Feature list tidak boleh kosong.")
        if len(features) != 5:
            raise ValueError(f"Dibutuhkan tepat 5 features, diterima {len(features)}.")
        if any(not isinstance(f, (int, float)) for f in features):
            raise ValueError("Semua feature harus berupa angka.")

        try:
            arr = np.array(features).reshape(1, -1)
            raw_prediction = self.model.predict(arr)[0]
            probability = float(self.model.predict_proba(arr)[0].max())
        except Exception as e:
            logger.error(f"Model inference gagal: {e}")
            raise RuntimeError(f"Inference gagal: {e}") from e

        tier = self._determine_tier(probability)

        result = {
            "prediction": int(raw_prediction),
            "probability": round(probability, 4),
            "risk_tier": tier,
            "features": features,
            "model_version": self.version
        }

        # --- Repository Persistence Block ---
        try:
            saved = self.repo.save(result)
            return saved
        except RuntimeError as e:
            # Mengamankan error database, bungkus dengan pesan baru yang deskriptif
            logger.error(f"Repository gagal menyimpan data: {e}")
            raise RuntimeError(f"Gagal menyimpan hasil prediksi ke database: {e}") from e

    def _determine_tier(self, probability: float) -> str:
        if probability < settings.RISK_LOW_MAX:
            return "low"
        elif probability < settings.RISK_MEDIUM_MAX:
            return "medium"
        return "high"