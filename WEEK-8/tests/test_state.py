# tests/test_state.py
from app.core.state import AppState, app_state

def test_app_state_default_values():
    """Verifikasi nilai default AppState dari blueprint aslinya."""
    state = AppState()
    
    assert state.model_service is None
    assert state.redis_client is None
    assert state.ready is False
    assert state.startup_error is None

def test_app_state_singleton_identity():
    """Verifikasi integritas pola Singleton pada memory address yang sama."""
    from app.core.state import app_state as state_a
    from app.core import state as state_module
    
    assert state_a is state_module.app_state

def test_ready_flag_mutation():
    """Test perubahan flag terdeteksi secara real-time di instance global."""
    assert app_state.ready is False
    
    app_state.ready = True
    assert app_state.ready is True