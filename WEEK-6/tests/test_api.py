"""
week-06/tests/test_api.py
W06D06 — ML API Local Testing

Jalankan dengan: python tests/test_api.py
Pastikan server sudah running: uvicorn app.main:app --reload
"""
import math
import httpx
import json


# ── CONFIG ────────────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:8000"
RESULTS = {"pass": 0, "fail": 0}

# ── HELPER ───────────────────────────────────────────────────

def run_test(
    name: str,
    method: str,
    endpoint: str,
    payload: dict | None = None,
    expected_status: int = 200,
) -> dict:
    """
    Kirim satu HTTP request, print hasilnya, dan assert status code.
    Return response body sebagai dict agar bisa di-inspect lebih lanjut.
    """
    url = f"{BASE_URL}{endpoint}"

    try:
        # httpx.request() bisa handle GET dan POST dengan argumen yang sama
        response = httpx.request(method, url, json=payload, timeout=10.0)
    except httpx.ConnectError:
        # Server belum running — beri pesan yang jelas
        print(f"\n[ERROR] Server tidak bisa dihubungi di {BASE_URL}")
        print("        Jalankan: uvicorn app.main:app --reload")
        raise SystemExit(1)

    # ── Print hasil ──────────────────────────────────────────
    status_ok = response.status_code == expected_status
    icon = "✓" if status_ok else "✗"
    print(f"\n{icon} {name}")
    print(f"  {method} {endpoint}")
    print(f"  Status : {response.status_code} (expected {expected_status})")

    # Pretty-print response body
    try:
        body = response.json()
        print(f"  Body   : {json.dumps(body, indent=2)[:300]}")  # truncate kalau panjang
    except Exception:
        print(f"  Body   : {response.text[:300]}")
        body = {}

    # ── Assert & track result ────────────────────────────────
    if status_ok:
        RESULTS["pass"] += 1
    else:
        RESULTS["fail"] += 1
        print(f"  !! GAGAL: expected {expected_status}, got {response.status_code}")

    return body


# ── TEST CASES ────────────────────────────────────────────────

def test_health_check():
    """GET /predict/health — server dan model harus ready."""
    body = run_test(
        name="Health check",
        method="GET",
        endpoint="/predict/health",
        expected_status=200,
    )
    # Cek isi body dari health endpoint kita
    assert body.get("status") == "ready", f"Expected status='ready', got: {body}"


def test_happy_path():
    """POST /predict/ dengan input valid — harus return 200 dan ada field 'prediction'."""
    payload = {
        "features": [5.1, 3.5, 1.4, 0.2]  # Format sesuai dengan n_features=4
    }
    body = run_test(
        name="Happy path — input valid",
        method="POST",
        endpoint="/predict/",
        payload=payload,
        expected_status=200,
    )
    assert "prediction" in body, f"Field 'prediction' tidak ada di response: {body}"


def test_empty_features():
    """POST /predict/ dengan array kosong — Pydantic harus return 422."""
    payload = {"features": []}
    run_test(
        name="Empty features → 422",
        method="POST",
        endpoint="/predict/",
        payload=payload,
        expected_status=422,
    )


def test_wrong_type():
    """POST /predict/ dengan string di dalam array — Pydantic harus return 422."""
    payload = {"features": ["bukan_angka", 3.5, 1.4, 0.2]}
    run_test(
        name="Wrong type (string in array) → 422",
        method="POST",
        endpoint="/predict/",
        payload=payload,
        expected_status=422,
    )


def test_missing_field():
    """POST /predict/ tanpa field 'features' sama sekali — Pydantic return 422."""
    payload = {"wrong_field": [1.0, 2.0]}
    run_test(
        name="Missing required field → 422",
        method="POST",
        endpoint="/predict/",
        payload=payload,
        expected_status=422,
    )


def test_nan_value():
    """
    POST /predict/ dengan NaN — domain validator (@field_validator) di schema
    harus menangkap ini dan return 422.
    (Kita kirim string "NaN" agar Pydantic membacanya sebagai representasi float NaN)
    """
    payload = {"features": ["NaN", 3.5, 1.4, 0.2]}
    run_test(
        name="NaN in features → 422 (field_validator)",
        method="POST",
        endpoint="/predict/",
        payload=payload,
        expected_status=422,
    )


def test_negative_value():
    """
    POST /predict/ dengan nilai negatif di index awal — domain logic di router
    harus return 400 (Bad Request), karena ini melanggar aturan bisnis, bukan skema.
    """
    payload = {"features": [-5.0, 3.5, 1.4, 0.2]}
    run_test(
        name="Negative value in first features → 400 (domain logic)",
        method="POST",
        endpoint="/predict/",
        payload=payload,
        expected_status=400,
    )


def test_wrong_feature_count():
    """
    POST /predict/ dengan jumlah fitur salah — domain logic router
    akan menolak ini dengan 400 Bad Request.
    """
    payload = {"features": [1.0, 2.0]}  # Hanya 2 fitur, sedangkan model expect 4
    run_test(
        name="Wrong feature count → 400 (domain logic)",
        method="POST",
        endpoint="/predict/",
        payload=payload,
        expected_status=400, 
    )


# ── MAIN ─────────────────────────────────────────────────────

def main() -> None:
    print("=" * 50)
    print("ML API — Local Test Suite")
    print("=" * 50)

    # Jalankan semua test
    test_health_check()
    test_happy_path()
    test_empty_features()
    test_wrong_type()
    test_missing_field()
    test_nan_value()
    test_negative_value()
    test_wrong_feature_count()

    # Ringkasan akhir
    total = RESULTS["pass"] + RESULTS["fail"]
    print("\n" + "=" * 50)
    print(f"Hasil: {RESULTS['pass']}/{total} pass")
    if RESULTS["fail"] > 0:
        print(f"       {RESULTS['fail']} test GAGAL — periksa output di atas")
    else:
        print("       Semua test passed ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()