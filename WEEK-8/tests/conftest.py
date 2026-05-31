# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.core.state import app_state

@pytest.fixture(autouse=True)
def reset_app_state():
    """
    Reset app_state ke kondisi default sebelum setiap test.
    Mencegah kebocoran state antar-test yang membuat hasil flaky.
    """
    app_state.model_service = None
    app_state.redis_client = None
    app_state.ready = False
    app_state.startup_error = None
    yield
    # teardown
    app_state.model_service = None
    app_state.redis_client = None
    app_state.ready = False
    app_state.startup_error = None

@pytest.fixture
def client():
    """
    TestClient yang berjalan TANPA mengeksekusi lifespan.
    Di FastAPI/Starlette modern, menginisialisasi TestClient tanpa blok `with`
    akan mengabaikan event startup/shutdown. Sangat cocok untuk test Liveness murni.
    """
    return TestClient(app)

@pytest.fixture
def ready_client():
    """
    Client dengan simulasi app_state sudah startup penuh.
    """
    app_state.ready = True
    
    # Mock redis_client agar ping() selalu sukses
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    app_state.redis_client = mock_redis
    
    # Mock model_service agar tidak None
    app_state.model_service = MagicMock()
    
    # Bypass pengecekan disk sebenarnya agar test tidak gagal jika harddisk lokal Anda penuh
    with patch("app.routers.health._check_disk_space", return_value=True):
        # Gunakan blok `with` agar middleware (seperti logging) tetap terinisiasi sempurna
        with TestClient(app) as c:
            yield c