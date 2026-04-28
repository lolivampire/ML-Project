import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path: str, version: str):
        self.model_path = model_path
        self.version = version
        self.model = None

    def load_model(self):
        try:
            # Load bundle dictionary
            bundle = joblib.load(self.model_path)
            
            # Ekstrak pipeline dan metadatanya
            self.model = bundle["pipeline"]
            
            # (Opsional) Kamu bisa pakai versi dari metadata model jika mau
            self.version = bundle["metadata"].get("version", self.version) 
            
            logger.info(f"Model successfully loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise e

    def predict(self, features: list[float]) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded yet.")

        X = np.array(features).reshape(1, -1)

        prediction = self.model.predict(X)[0]

        probability = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            probability = float(max(proba))

        return {
            "prediction": int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction),
            "probability": probability,
            "model_version": self.version
        }