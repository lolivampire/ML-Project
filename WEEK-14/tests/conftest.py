import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

# Import aplikasi utama kita
from main import app

@pytest_asyncio.fixture
async def async_client():
    """
    Fixture ini membuat HTTP Client virtual yang langsung menembak
    ke lapisan ASGI FastAPI secara asinkron.
    """
    transport = ASGITransport(app=app)
    # base_url "http://test" hanya sebagai domain dummy untuk testing
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client