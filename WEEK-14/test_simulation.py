# test_simulation.py
from app.schemas.input_schema import DecisionInput, MarketCondition, RiskLevel, TimeHorizon, IndustrySector
from app.services.simulation_services import SimulationService

# 1. Buat input valid (sesuai schema yang sudah direfaktor)
# Perhatikan perubahan nama field dan penggunaan Enum agar lolos Pydantic
data = DecisionInput(
    company_name="Lolivampire Corp",
    budget=50_000_000.0,
    growth_target=3.0,                  # Pengganti target_revenue (Misal: ingin 3x lipat dari budget)
    market_condition=MarketCondition.BULL, # Sebelumnya BULLISH, sekarang BULL
    risk_appetite=RiskLevel.MEDIUM,        # Sebelumnya risk_tolerance
    time_horizon=TimeHorizon.MEDIUM,       # Sebelumnya time_horizon_months
    industry_sector=IndustrySector.TECHNOLOGY # Wajib diisi sesuai kontrak Pydantic
)

# 2. Inisialisasi Service dan Eksekusi
service = SimulationService()
result = service.run_simulation(data)

# 3. Cetak Hasil
print("=== SIMULATION OUTPUT ===")
print(f"Optimistic  -> score: {result.optimistic.score:.1f}, revenue: Rp {result.optimistic.projected_revenue:,.0f}")
print(f"Realistic   -> score: {result.realistic.score:.1f}, revenue: Rp {result.realistic.projected_revenue:,.0f}")
print(f"Pessimistic -> score: {result.pessimistic.score:.1f}, revenue: Rp {result.pessimistic.projected_revenue:,.0f}")
print(f"\nSummary: {result.summary}")