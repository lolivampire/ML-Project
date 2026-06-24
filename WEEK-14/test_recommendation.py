# test_recommendation.py
from pydantic import ValidationError
from app.schemas.input_schema import DecisionInput, MarketCondition, RiskLevel, TimeHorizon, IndustrySector
from app.services.simulation_services import SimulationService
from app.services.recommendation_service import RecommendationService

# 1. Inisialisasi Service (Dependency Injection manual untuk testing)
sim_svc = SimulationService()
rec_svc = RecommendationService()

# 2. Definisikan 3 variasi profil risiko yang ingin diuji
test_risks = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]

print("=== 1. PENGUJIAN REKOMENDASI BERDASARKAN PROFIL RISIKO ===\n")

for risk in test_risks:
    # 3. Buat Payload Dinamis (hanya risk_appetite yang berubah di setiap iterasi)
    payload = DecisionInput(
        company_name="Nusantara Tech",
        budget=50_000_000.0,
        growth_target=2.0,                  # Target 2x lipat
        risk_appetite=risk,                 # <-- Variabel yang disuntikkan
        market_condition=MarketCondition.NEUTRAL,
        time_horizon=TimeHorizon.MEDIUM,
        industry_sector=IndustrySector.TECHNOLOGY
    )

    # 4. Eksekusi Orkestrasi Service
    sim_output = sim_svc.run_simulation(payload)
    rec_output = rec_svc.recommend(sim_output, payload)

    # 5. Cetak Hasil Tertarget
    print(f"MENGUJI PROFIL: {risk.name}")
    print(f"Rekomendasi   : Skenario {rec_output.recommended_scenario.upper()}")
    print(f"Proyeksi      : Rp {rec_output.projected_revenue:,.0f} (Keyakinan: {rec_output.confidence:.0%})")
    print(f"Reasoning     :\n{rec_output.reasoning}")
    print("-" * 80)


print("=== 2. PENGUJIAN JALUR GAGAL (EDGE CASE: EXTREME) ===")
try:
    # Kita sengaja menyuntikkan string "extreme" yang tidak terdaftar di Enum
    bad_payload = DecisionInput(
        company_name="Nusantara Tech",
        budget=50_000_000.0,
        growth_target=2.0,
        risk_appetite="extreme",  # <-- KESENGAJAAN ERROR DI SINI
        market_condition=MarketCondition.NEUTRAL,
        time_horizon=TimeHorizon.MEDIUM,
        industry_sector=IndustrySector.TECHNOLOGY
    )
    
    # Jika Pydantic kebobolan, baris di bawah ini akan tereksekusi
    print("[BAHAYA] Data kotor berhasil masuk ke sistem!")
    sim_svc.run_simulation(bad_payload)

except ValidationError as e:
    # Pydantic menangkap kesalahan sebelum objek DecisionInput selesai dibuat
    print("[DITOLAK] Pertahanan bekerja! Pydantic menolak data sebelum menyentuh Service.")
    
    # Membedah isi pesan error dari Pydantic
    for err in e.errors():
        field_error = err['loc'][0]
        pesan_error = err['msg']
        print(f" -> Detail Error pada field '{field_error}': {pesan_error}")