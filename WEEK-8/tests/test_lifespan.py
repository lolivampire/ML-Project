# tests/test_lifespan.py
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.state import app_state
from app.core.lifespan import lifespan

def app_with_lifespan():
    """Helper: buat fresh FastAPI app dengan lifespan."""
    return FastAPI(lifespan=lifespan)

def test_ready_is_last_step_in_startup():
    """
    app_state.ready harus menjadi baris eksekusi TERAKHIR saat startup.
    Mencegah Load Balancer mengirim request sebelum model/DB siap.
    """
    call_order = []
    
    mock_redis = MagicMock()
    mock_model_instance = MagicMock()
    
    # Patch target disesuaikan dengan isi app/core/lifespan.py
    with patch("app.core.lifespan.ModelService", return_value=mock_model_instance), \
         patch("app.core.lifespan.create_redis_client", return_value=mock_redis):
        
        original_setattr = app_state.__class__.__setattr__
        
        def tracking_setattr(self, name, value):
            if name == "ready" and value is True:
                call_order.append("ready_set_true")
            elif name == "model_service" and value is not None:
                call_order.append("model_loaded")
            elif name == "redis_client" and value is not None:
                call_order.append("redis_set")
            original_setattr(self, name, value)
        
        with patch.object(app_state.__class__, "__setattr__", tracking_setattr):
            with TestClient(app_with_lifespan()):
                pass  # Proses startup terjadi ketika memasuki context
    
    assert "ready_set_true" in call_order
    ready_index = call_order.index("ready_set_true")
    
    for step in ["model_loaded", "redis_set"]:
        assert step in call_order
        assert call_order.index(step) < ready_index, f"'{step}' harus tereksekusi sebelum 'ready_set_true'"


def test_ready_first_in_shutdown():
    """
    app_state.ready harus bernilai False PERTAMA KALI saat shutdown.
    Mencegah traffic masuk ke ruang memori yang sedang dibongkar.
    """
    shutdown_order = []
    
    mock_redis = MagicMock()
    mock_model_instance = MagicMock()
    
    with patch("app.core.lifespan.ModelService", return_value=mock_model_instance), \
         patch("app.core.lifespan.create_redis_client", return_value=mock_redis):
        
        original_setattr = app_state.__class__.__setattr__
        
        def tracking_setattr(self, name, value):
            if name == "ready" and value is False:
                shutdown_order.append("ready_set_false")
            original_setattr(self, name, value)
        
        def tracking_close():
            shutdown_order.append("redis_closed")
            
        mock_redis.close = tracking_close
        
        with patch.object(app_state.__class__, "__setattr__", tracking_setattr):
            # Memasuki lalu langsung keluar context akan memicu startup + shutdown
            with TestClient(app_with_lifespan()):
                pass 
    
    assert "ready_set_false" in shutdown_order
    assert "redis_closed" in shutdown_order
    assert shutdown_order.index("ready_set_false") < shutdown_order.index("redis_closed"), \
           "ready=False harus dieksekusi SEBELUM redis.close()"