# app/services/recommendation_service.py
from app.schemas.input_schema import DecisionInput, RiskLevel
from app.schemas.simulation_schema import SimulationOutput, RecommendationOutput

class RecommendationService:
    """
    Service layer murni — tidak menyentuh lapisan HTTP.
    Mengevaluasi hasil simulasi matematis dan mengonversinya menjadi 
    keputusan bisnis yang dapat ditindaklanjuti berdasarkan profil risiko.
    """

    def recommend(
        self,
        simulation: SimulationOutput,
        input_data: DecisionInput,
    ) -> RecommendationOutput:
        """
        Mengekstrak skenario paling relevan dan membungkusnya dengan justifikasi.
        """
        
        # 1. Mapping risk_appetite (Enum) -> nama skenario
        scenario_map = {
            RiskLevel.LOW: "pessimistic",
            RiskLevel.MEDIUM: "realistic",
            RiskLevel.HIGH: "optimistic",
        }

        # Mengambil nama skenario target
        chosen_name = scenario_map[input_data.risk_appetite]

        # 2. Ambil objek SimulationResult langsung dari atribut SimulationOutput
        # Karena atributnya bernama "optimistic", "realistic", dsb., kita bisa pakai getattr
        chosen_scenario = getattr(simulation, chosen_name)

        # 3. Bangun reasoning string
        reasoning = self._build_reasoning(
            scenario_name=chosen_name,
            risk_appetite=input_data.risk_appetite.value,
            confidence=chosen_scenario.confidence,
            projected_revenue=chosen_scenario.projected_revenue,
        )

        # 4. Kembalikan output yang sudah dikemas rapat
        return RecommendationOutput(
            recommended_scenario=chosen_name,
            confidence=round(chosen_scenario.confidence, 2),
            projected_revenue=round(chosen_scenario.projected_revenue, 2),
            reasoning=reasoning,
            simulation_details=simulation,  # Menyertakan seluruh 3 skenario + summary
        )

    def _build_reasoning(
        self,
        scenario_name: str,
        risk_appetite: str,
        confidence: float,
        projected_revenue: float,
    ) -> str:
        """
        Menyusun narasi rekomendasi.
        Private method — diisolasi hanya untuk keperluan di dalam service ini.
        """
        templates = {
            "pessimistic": (
                f"Profil risiko entitas Anda tergolong konservatif (Level: {risk_appetite}). "
                f"Skenario 'Pessimistic' diprioritaskan untuk memastikan ekspektasi Anda "
                f"terlindungi secara maksimal dari guncangan kondisi pasar terburuk. "
                f"Proyeksi pendapatan: Rp {projected_revenue:,.0f} dengan tingkat keyakinan {confidence:.0%}."
            ),
            "realistic": (
                f"Profil risiko entitas Anda tergolong moderat (Level: {risk_appetite}). "
                f"Skenario 'Realistic' menggunakan asumsi median yang paling sering akurat "
                f"berdasarkan data historis industri. "
                f"Proyeksi pendapatan: Rp {projected_revenue:,.0f} dengan tingkat keyakinan {confidence:.0%}."
            ),
            "optimistic": (
                f"Profil risiko entitas Anda tergolong agresif (Level: {risk_appetite}). "
                f"Skenario 'Optimistic' memaksimalkan potensi pertumbuhan dengan mengasumsikan "
                f"kondisi eksekusi yang sempurna. "
                f"Proyeksi pendapatan: Rp {projected_revenue:,.0f} dengan tingkat keyakinan {confidence:.0%}."
            ),
        }
        return templates[scenario_name]