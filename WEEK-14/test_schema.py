# test_schema.py
from app.schemas.simulation_schema import SimulationResult
from pydantic import ValidationError

print("=== TEST 1: Instansiasi Jalur Sukses (Happy Path) ===")
try:
    r = SimulationResult(
        scenario="optimistic",
        score=85.5,
        projected_revenue=1500000.0,
        risk_level="low",
        recommendation="Lanjutkan eksekusi.",
        confidence=0.60,
    )
    print("    SimulationResult OK!")
    print(f"   Skenario : {r.scenario}")
    print(f"   Skor     : {r.score}")
except Exception as e:
    print(f"❌ Gagal secara tak terduga: {e}")

print("\n=== TEST 2: Validasi Literal Pydantic (Jalur Gagal) ===")
try:
    bad = SimulationResult(
        scenario="unknown",       # ← Error 1: Bukan "optimistic", "realistic", atau "pessimistic"
        score=50.0,
        projected_revenue=100.0,
        risk_level="extreme",     # ← Error 2: Bukan "low", "medium", atau "high"
        recommendation="test",
        confidence=0.5,
    )
    print("❌ SALAH — Pydantic kebobolan (seharusnya raise ValidationError)!")
except ValidationError as e:
    print("    Literal validation bekerja dengan sempurna!")
    print(f"   Jumlah error tertangkap: {len(e.errors())}")
    
    # Membedah error untuk melihat detail dari Pydantic
    for i, err in enumerate(e.errors(), 1):
        field_name = err["loc"][0]
        error_msg = err["msg"]
        print(f"   - Error {i} pada field '{field_name}': {error_msg}")