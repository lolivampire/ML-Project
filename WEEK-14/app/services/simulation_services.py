# app/services/simulation_service.py
import math
from app.schemas.input_schema import DecisionInput, MarketCondition, RiskLevel
from app.schemas.simulation_schema import SimulationResult, SimulationOutput

class SimulationService:
    """
    Service layer yang menangani orkestrasi pemodelan keputusan.
    Berfungsi murni sebagai mesin logika bisnis tanpa menyentuh layer HTTP.
    """

    def run_simulation(self, data: DecisionInput) -> SimulationOutput:
        """Entry point utama untuk menghasilkan tiga skenario simulasi."""
        base_score = self._calculate_base_score(data)
        base_revenue = self._calculate_base_revenue(data)

        optimistic  = self._build_scenario(data, base_score, base_revenue, "optimistic")
        realistic   = self._build_scenario(data, base_score, base_revenue, "realistic")
        pessimistic = self._build_scenario(data, base_score, base_revenue, "pessimistic")

        summary = self._generate_summary(optimistic, realistic, pessimistic)

        return SimulationOutput(
            optimistic=optimistic,
            realistic=realistic,
            pessimistic=pessimistic,
            summary=summary,
        )

    # ── PRIVATE METHODS ──────────────────────────────────────────

    def _calculate_base_score(self, data: DecisionInput) -> float:
        """Menghitung skor probabilitas awal berdasarkan input pengguna."""
        score = 50.0  # Baseline netral

        # Peningkatan skor logaritmik berdasarkan budget
        if data.budget > 0:
            score += min(20.0, math.log10(data.budget) * 3)

        # Modifikasi berdasarkan kondisi pasar (Sesuai Enum MarketCondition)
        market_bonus = {
            MarketCondition.BULL: 15.0,
            MarketCondition.NEUTRAL: 0.0,
            MarketCondition.BEAR: -15.0,
        }
        score += market_bonus.get(data.market_condition, 0.0)

        # Modifikasi berdasarkan toleransi risiko (Sesuai Enum RiskLevel)
        risk_bonus = {
            RiskLevel.LOW: -5.0,
            RiskLevel.MEDIUM: 0.0,
            RiskLevel.HIGH: 8.0,
        }
        score += risk_bonus.get(data.risk_appetite, 0.0)

        return max(0.0, min(100.0, score))

    def _calculate_base_revenue(self, data: DecisionInput) -> float:
        """
        Menghitung proyeksi pendapatan baseline.
        Karena tidak ada input target_revenue, kita mengkalikan budget dengan growth_target.
        """
        target_revenue = data.budget * data.growth_target
        return target_revenue * 0.8  # Set 80% dari target sebagai baseline realistis

    def _build_scenario(
        self,
        data: DecisionInput,
        base_score: float,
        base_revenue: float,
        scenario_type: str,
    ) -> SimulationResult:
        """Membangun matriks output individual berdasarkan tipe skenario."""
        config = {
            "optimistic":  {"score_mult": 1.3,  "rev_mult": 1.5,  "risk": "low",    "conf": 0.60},
            "realistic":   {"score_mult": 1.0,  "rev_mult": 1.0,  "risk": "medium", "conf": 0.80},
            "pessimistic": {"score_mult": 0.65, "rev_mult": 0.6,  "risk": "high",   "conf": 0.70},
        }
        cfg = config[scenario_type]

        score      = min(100.0, base_score * cfg["score_mult"])
        revenue    = base_revenue * cfg["rev_mult"]
        risk       = cfg["risk"]
        confidence = cfg["conf"]

        # Penalti tambahan untuk skenario pesimis jika user sangat agresif
        if scenario_type == "pessimistic" and data.risk_appetite == RiskLevel.HIGH:
            revenue *= 0.85

        recommendation = self._generate_recommendation(scenario_type, score, revenue)

        return SimulationResult(
            scenario=scenario_type,
            score=round(score, 2),
            projected_revenue=round(revenue, 2),
            risk_level=risk,
            recommendation=recommendation,
            confidence=confidence,
        )

    def _generate_recommendation(self, scenario: str, score: float, revenue: float) -> str:
        """Menghasilkan teks rekomendasi otomatis berdasarkan parameter output."""
        if scenario == "optimistic":
            return (
                f"Dalam kondisi terbaik, proyeksi pendapatan mencapai "
                f"Rp {revenue:,.0f} dengan skor keberhasilan {score:.1f}/100. "
                f"Rekomendasi: lanjutkan dengan eksekusi penuh dan monitoring ketat."
            )
        elif scenario == "realistic":
            return (
                f"Dengan asumsi kondisi rata-rata, proyeksi pendapatan "
                f"Rp {revenue:,.0f} dan skor {score:.1f}/100. "
                f"Rekomendasi: set milestone bulanan dan review di minggu ke-6."
            )
        else:
            return (
                f"Dalam skenario terburuk, pendapatan hanya mencapai "
                f"Rp {revenue:,.0f} dengan skor {score:.1f}/100. "
                f"Rekomendasi: siapkan contingency plan dan exit criteria yang jelas."
            )

    def _generate_summary(self, opt: SimulationResult, real: SimulationResult, pess: SimulationResult) -> str:
        """Menyusun ringkasan eksekutif pembanding spread."""
        spread = opt.projected_revenue - pess.projected_revenue
        return (
            f"Range proyeksi: Rp {pess.projected_revenue:,.0f} – Rp {opt.projected_revenue:,.0f} "
            f"(spread Rp {spread:,.0f}). Skenario realistis menunjukkan skor {real.score:.1f}/100."
        )