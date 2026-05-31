# 🎯 Risk Scoring API — Containerized & Orchestration-Ready

> Production-grade ML API featuring Redis caching, multi-stage Docker builds,
> structured JSON logging, and enterprise lifecycle management.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-blue?logo=docker)](https://docker.com)
[![Redis](https://img.shields.io/badge/Redis-Caching-red?logo=redis)](https://redis.io)
[![Tests](https://img.shields.io/badge/Tests-Passing-success?logo=pytest)](./tests)

## About

This API evaluates credit risk probability from applicant financial features
using a trained scikit-learn pipeline. Built as **Project 2** of a 24-week
ML Engineering Roadmap, this iteration transitions from a single-container
deployment into a robust, orchestration-ready distributed system.

Key additions over Project 1: Redis caching layer, structured JSON logging
with per-request tracing IDs, multi-stage Docker builds, and rigorous
lifecycle management designed to integrate with Kubernetes or Docker Swarm.

## Architecture

### Layer structure

```
Request
   │
   ▼
middleware/logging_middleware.py   ← injects request_id, tracks duration
   │
   ▼
routers/predict.py  ·  routers/health.py   ← input validation, routing
   │
   ▼
services/model_service.py          ← ML inference logic
   │
   ▼
database.py  ·  core/state.py      ← Redis + AppState singleton
```

### Lifecycle management

Project 2 uses FastAPI's `lifespan` context manager to enforce strict
startup and shutdown ordering. `app_state.ready` acts as a traffic gate —
it is the **last** flag set on startup and the **first** cleared on shutdown.

| Phase | Order | Action |
|-------|-------|--------|
| Startup | 1 | Load ML model → `app_state.model_service` |
| Startup | 2 | Establish Redis connection → `app_state.redis_client` |
| Startup | 3 (last) | `app_state.ready = True` — opens traffic gateway |
| Shutdown | 1 (first) | `app_state.ready = False` — drops from load balancer |
| Shutdown | 2 | Drain in-flight requests |
| Shutdown | 3 | Close Redis connection gracefully |
| Shutdown | 4 | Unload model and exit process |

### Health vs readiness probes

| Probe | Endpoint | Purpose | Checks dependency? | Consequence of failure |
|-------|----------|---------|-------------------|----------------------|
| Liveness | `GET /health` | Confirms Python process is alive and not deadlocked | ❌ None | Orchestrator sends SIGKILL → container restarts |
| Readiness | `GET /ready` | Confirms API can process requests | ✅ Redis ping, model state, disk space | Traffic removed from load balancer — container stays alive |

## Folder structure

```
ML-Project/
├── WEEK-8/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py             # Pydantic BaseSettings — env var management
│   │   │   ├── logger.py             # JSON formatter + request ID tracing
│   │   │   ├── state.py              # AppState singleton (model, redis, flags)
│   │   │   └── lifespan.py           # Startup/shutdown context manager
│   │   ├── middleware/
│   │   │   └── logging_middleware.py # Injects UUID, tracks duration & status
│   │   ├── routers/
│   │   │   ├── predict.py            # POST /predict
│   │   │   └── health.py             # GET /health + GET /ready
│   │   ├── services/
│   │   │   └── model_service.py      # ML model loading and inference
│   │   ├── database.py               # Redis connection + retry logic
│   │   └── main.py                   # FastAPI app orchestration
│   ├── tests/
│   │   ├── conftest.py               # Shared fixtures, mock clients, autouse reset
│   │   ├── test_health.py            # Liveness and readiness logic
│   │   ├── test_lifespan.py          # Startup/shutdown execution order
│   │   └── test_state.py             # Singleton integrity checks
│   ├── .env.dev                      # Environment variables (development)
│   ├── docker-compose.yml            # Base compose (infrastructure)
│   ├── docker-compose.dev.yml        # Dev overrides (ports, volumes, hot reload)
│   ├── Dockerfile                    # Multi-stage build definition
│   ├── requirements.txt              # Python dependencies
│   └── README.md
```

## Quick Start

**Prerequisites:** Docker >= 24.0 · Docker Compose >= 2.0

### Path 1: Docker Compose (recommended)

Spins up both the FastAPI application and a dedicated Redis instance
in an isolated Docker network.

```bash
git clone https://github.com/lolivampire/ML-Project
cd ML-Project/WEEK-8

# Build and start all services
docker-compose --env-file .env.dev \
  -f docker-compose.yml \
  -f docker-compose.dev.yml \
  up --build
```

API: `http://localhost:8000` | Docs: `http://localhost:8000/docs`

### Path 2: Local virtual environment

Use this path for active development. Requires a running Redis server locally.

```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate        # Windows PowerShell
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Set environment variables (PowerShell)
$env:APP_ENV="development"
$env:DEBUG="true"
$env:REDIS_HOST="localhost"

# Start the server
uvicorn app.main:app --reload
```

## API Reference

### POST /predict

Executes risk scoring prediction. Request payload and result are logged
in structured JSON with a unique `request_id` for tracing.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "income": 15000000,
    "age": 28,
    "debt_ratio": 0.35
  }'
```

**Response:**
```json
{
  "risk_score": 0.24,
  "decision": "APPROVED",
  "model_version": "v1.0.0"
}
```

**Response legend:**
- `risk_score`: probability of default (0.0 – 1.0)
- `decision`: `"APPROVED"` or `"REJECTED"` based on threshold
- `model_version`: model identifier for traceability

---

### GET /health

Liveness probe. Confirms the server process is running — does not check
any external dependency. Always returns `200 OK` while the process is alive.

**Response:**
```json
{
  "status": "alive",
  "uptime_seconds": 345.12,
  "version": "1.0.0"
}
```

---

### GET /ready

Readiness probe. Actively checks all critical dependencies before accepting
traffic. Returns `200 OK` when fully operational, `503 Service Unavailable`
if any check fails.

**Response — 200 OK (all checks pass):**
```json
{
  "status": "ready",
  "checks": {
    "model_loaded": true,
    "database_connected": true,
    "disk_space_ok": true
  }
}
```

**Response — 503 Service Unavailable (Redis unreachable):**
```json
{
  "status": "not_ready",
  "checks": {
    "model_loaded": true,
    "database_connected": false,
    "disk_space_ok": true
  }
}
```

## Testing

The project uses `pytest` with extensive `MagicMock` to ensure test
reliability without requiring live infrastructure.

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=term-missing
```

### Test output

```
tests/test_health.py::TestHealthEndpoint::test_health_always_200 PASSED                                                                                                                                        [  6%]
tests/test_health.py::TestHealthEndpoint::test_health_response_structure PASSED                                                                                                                                [ 13%]
tests/test_health.py::TestHealthEndpoint::test_health_not_affected_by_ready_state PASSED                                                                                                                       [ 20%]
tests/test_health.py::TestHealthEndpoint::test_health_returns_app_version PASSED                                                                                                                               [ 26%]
tests/test_health.py::TestReadyEndpoint::test_ready_returns_503_when_not_ready PASSED                                                                                                                          [ 33%]
tests/test_health.py::TestReadyEndpoint::test_ready_returns_200_when_fully_ready PASSED                                                                                                                        [ 40%]
tests/test_health.py::TestReadyEndpoint::test_ready_503_when_redis_ping_fails PASSED                                                                                                                           [ 46%]
tests/test_health.py::TestReadyEndpoint::test_ready_response_includes_all_checks PASSED                                                                                                                        [ 53%]
tests/test_health.py::TestReadyEndpoint::test_ready_503_response_body_explains_why PASSED                                                                                                                      [ 60%]
tests/test_health.py::TestReadyEndpoint::test_ready_503_does_not_restart_container PASSED                                                                                                                      [ 66%]
tests/test_lifespan.py::test_ready_is_last_step_in_startup PASSED                                                                                                                                              [ 73%]
tests/test_lifespan.py::test_ready_first_in_shutdown PASSED                                                                                                                                                    [ 80%]
tests/test_state.py::test_app_state_default_values PASSED                                                                                                                                                      [ 86%]
tests/test_state.py::test_app_state_singleton_identity PASSED                                                                                                                                                  [ 93%]
tests/test_state.py::test_ready_flag_mutation PASSED                                                                                                                                                           [100%]

15 passed in 0.24s
```


### Coverage by layer

| Layer | File | What is tested |
|-------|------|----------------|
| State management | `test_state.py` | Singleton pattern — same object across imports; no state leakage between tests via `autouse` fixture |
| Health probes | `test_health.py` | `/health` unaffected by Redis failure; `/ready` correctly returns 503 on disconnection or missing model |
| Lifecycle ordering | `test_lifespan.py` | `app_state.ready = True` is set **last** on startup; `app_state.ready = False` is set **first** on shutdown — verified via `call_order` list |

## Docker

### Multi-stage build

Separates the build environment (compiling C-extensions for numpy, pandas,
scikit-learn) from the lean production runtime.

```dockerfile
# Stage 1 — builder: compile all dependencies
FROM python:3.11-slim AS builder
RUN pip install --user -r requirements.txt

# Stage 2 — production: copy only compiled binaries
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY app/ ./app/
```

| Stage | Base image | Contents | Approx. size |
|-------|-----------|----------|-------------|
| Builder | `python:3.11-slim` | gcc, g++, pip cache, build tools | ~900 MB |
| Production | `python:3.11-slim` | Compiled binaries + app code only | ~180 MB |

### Build and run (standalone)

```bash
docker build -t risk-scoring-api:v2 .
docker run -d -p 8000:8000 --name risk-api risk-scoring-api:v2
```

### Useful commands

```bash
docker-compose down -v               # Stop all containers, wipe Redis volume
docker logs -f risk_api              # Tail application logs in real-time
docker exec -it risk_redis redis-cli # Enter Redis interactive terminal
docker images risk-scoring-api       # Check image size before/after build
docker volume prune                  # Clean up unused volumes
```

## Limitations & Known Issues

- **Single-node architecture:** State management and model loading assume one
  API instance. Horizontal scaling requires shared storage (e.g., AWS S3)
  for model distribution across nodes.
- **No authentication:** All endpoints are public. Not suitable for PII data
  without adding an auth middleware layer (e.g., JWT or API key validation).
- **Redis standalone:** Configured as a single instance without persistence
  or clustering — cached data is volatile on container destruction.
- **Disk check scope:** The `/ready` disk space check measures the container's
  root filesystem. In Kubernetes with persistent volume mounts, this may
  reflect node capacity rather than actual PVC limits.
- **No request rate limiting:** High concurrent load is not throttled beyond
  Pydantic input validation.

---

*Project 2 of 24 — ML Engineering Roadmap · [github.com/lolivampire/ML-Project](https://github.com/lolivampire/ML-Project)*