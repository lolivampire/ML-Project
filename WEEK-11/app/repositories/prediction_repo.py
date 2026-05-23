import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class PredictionRepository:
    """
    Bertanggung jawab menyimpan dan mengambil data prediksi.
    Menerima injeksi data store dari luar.
    """
    def __init__(self, store: List[Dict[str, Any]]):
        # Menerima list dari app.state (dikelola oleh framework FastAPI)
        self._store = store

    def save(self, prediction_data: dict) -> dict:
        record = {
            **prediction_data,
            "id": len(self._store) + 1,
            "created_at": datetime.utcnow().isoformat(),
        }
        self._store.append(record)
        logger.debug(f"Prediksi disimpan: id={record['id']}")
        return record

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._store)

    def get_by_id(self, prediction_id: int) -> Optional[dict]:
        for record in self._store:
            if record["id"] == prediction_id:
                return record
        return None