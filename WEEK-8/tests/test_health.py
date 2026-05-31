# tests/test_health.py
import pytest
import redis
from unittest.mock import MagicMock

from app.core.state import app_state
from app.config import settings

class TestHealthEndpoint:
    """Test /health — Liveness probe."""

    def test_health_always_200(self, client):
        """/health harus return 200 meskipun semua dependency mati."""
        app_state.ready = False
        app_state.redis_client = None
        app_state.model_service = None
        
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Verifikasi struktur JSON liveness."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "alive"  # Disesuaikan dari "ok" ke "alive"
        assert "uptime_seconds" in data

    def test_health_not_affected_by_ready_state(self, client):
        """/health sama sekali tidak peduli dengan flag readiness."""
        app_state.ready = False
        assert client.get("/health").status_code == 200
        
        app_state.ready = True
        assert client.get("/health").status_code == 200
    
    def test_health_returns_app_version(self, client):
        """
        Tujuan: /health response harus include field "version".
        Sangat berguna untuk CI/CD pipeline memastikan versi image Docker yang
        ter-deploy sudah sesuai dengan yang diharapkan.
        """
        response = client.get("/health")
        data = response.json()
        
        # Harus ada key 'version'
        assert "version" in data
        # Nilainya harus berupa string dan cocok dengan setting aplikasi
        assert isinstance(data["version"], str)
        assert data["version"] == settings.app_version


class TestReadyEndpoint:
    """Test /ready — Readiness probe."""

    def test_ready_returns_503_when_not_ready(self, client):
        """Return 503 saat app belum selesai startup (ready=False)."""
        app_state.ready = False
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"

    def test_ready_returns_200_when_fully_ready(self, ready_client):
        """Return 200 saat semua komponen sehat."""
        response = ready_client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_ready_503_when_redis_ping_fails(self, client):
        """Return 503 saat Redis mengalami runtime disconnection."""
        app_state.ready = True
        app_state.model_service = MagicMock()
        
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = redis.RedisError("Connection timeout")
        app_state.redis_client = mock_redis
        
        response = client.get("/ready")
        assert response.status_code == 503
        
        data = response.json()
        assert data["checks"]["database_connected"] is False

    def test_ready_response_includes_all_checks(self, ready_client):
        """Memastikan semua checks diobservasi (Model, Redis, Disk)."""
        response = ready_client.get("/ready")
        data = response.json()
        
        assert "checks" in data
        checks = data["checks"]
        
        # Harus sesuai dengan key yang kita definisikan di health.py
        assert "model_loaded" in checks
        assert "database_connected" in checks
        assert "disk_space_ok" in checks
    
    def test_ready_503_response_body_explains_why(self, client):
        """
        Tujuan: Saat /ready return 503, response body harus punya field yang menjelaskan
        KENAPA tidak ready (bukan hanya status code).
        Test ini memaksa endpoint untuk informatif.
        """
        # Kita set aplikasi seolah-olah "ready", tapi kita sengaja
        # mengosongkan model_service agar simulasi load model gagal.
        app_state.ready = True
        app_state.model_service = None
        
        # Mock Redis agar sukses (kita hanya ingin mengetes kegagalan model)
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        app_state.redis_client = mock_redis
        
        response = client.get("/ready")
        
        # Harus mengembalikan 503
        assert response.status_code == 503
        data = response.json()
        
        # Harus memiliki key yang menjelaskan alasannya (di desain kita, kita pakai "checks")
        assert "checks" in data
        
        # Pemeriksaan informatif: Harus jelas tertulis bahwa model_loaded adalah penyebab False-nya
        assert data["checks"]["model_loaded"] is False
        assert data["checks"]["database_connected"] is True

    def test_ready_503_does_not_restart_container(self, client):
        """
        Verifikasi bahwa /ready 503 TIDAK menyebabkan proses Python crash.
        
        Ini bukan test behavior — ini test bahwa endpoint tidak throw
        unhandled exception saat return 503. Ada perbedaan antara:
        - raise HTTPException(503) ← ini benar, FastAPI handle dengan clean
        - raise Exception("something") ← ini crash, container restart
        
        TestClient dengan raise_server_exceptions=True akan expose
        perbedaan ini.
        """
        app_state.ready = False
        
        # Ini harus tidak raise exception apapun di sisi Python
        # Kalau raise, berarti kamu pakai bare Exception bukan HTTPException
        try:
            response = client.get("/ready")
            assert response.status_code == 503
        except Exception as e:
            pytest.fail(f"/ready raise exception yang tidak seharusnya: {e}")