# app/frontend/streamlit_app.py
"""
Decision Support System — Streamlit Frontend
Menyediakan antarmuka interaktif untuk mesin simulasi berbasis FastAPI.
"""

import streamlit as st
import requests
import pandas as pd

# ── CONFIG ──────────────────────────────────────────────────────────
API_BASE_URL = "http://127.0.0.1:8000/api/v1/decision"

# ── PAGE SETUP ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="DSS Engine | Simulation",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧭 Decision Support System")
st.caption("Mesin Proyeksi Finansial dan Analisis Skenario Berbasis Risiko")

# ── SESSION STATE INIT ──────────────────────────────────────────────
if "api_result" not in st.session_state:
    st.session_state.api_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None

# ── SIDEBAR: INPUT FORM ─────────────────────────────────────────────
with st.sidebar:
    st.header("Parameter Input")
    st.divider()

    # --- Profil Entitas ---
    st.subheader("Profil Entitas")
    company_name = st.text_input("Nama Perusahaan", value="Nusantara Tech", max_chars=100)
    industry_sector = st.selectbox(
        "Sektor Industri", 
        options=["technology", "finance", "healthcare", "retail", "other"], 
        index=0
    )

    st.divider()

    # --- Parameter Finansial ---
    st.subheader("Finansial Dasar")
    budget = st.number_input(
        "Total Anggaran (Rp)",
        min_value=50000.0,  # Mematuhi aturan bisnis minimum 50.000
        value=50000000.0,
        step=5000000.0,
        format="%f"
    )
    growth_target = st.number_input(
        "Target Pertumbuhan (x Multiplier)",
        min_value=1.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Misal: 2.0 berarti target pendapatan adalah 2x lipat dari anggaran."
    )

    st.divider()

    # --- Profil Risiko & Pasar ---
    st.subheader("Kondisi & Risiko")
    risk_appetite = st.selectbox(
        "Toleransi Risiko",
        options=["low", "medium", "high"],
        index=1,
        format_func=lambda x: {"low": " Rendah (Konservatif)", "medium": " Menengah (Moderat)", "high": " Tinggi (Agresif)"}[x]
    )
    market_condition = st.selectbox(
        "Sentimen Pasar Saat Ini",
        options=["bull", "neutral", "bear"],
        index=1,
        format_func=lambda x: {"bull": " Bull (Naik)", "neutral": " Neutral (Datar)", "bear": " Bear (Turun)"}[x]
    )
    time_horizon = st.selectbox(
        "Horizon Waktu",
        options=["short", "medium", "long"],
        index=1
    )

    st.divider()

    # --- Submit button ---
    submitted = st.button("🔍 Jalankan Simulasi", use_container_width=True, type="primary")

# ── HTTP CLIENT / API CALL ──────────────────────────────────────────
if submitted:
    # Membangun payload sesuai skema DecisionInput Pydantic
    payload = {
        "company_name": company_name,
        "budget": budget,
        "growth_target": growth_target,
        "risk_appetite": risk_appetite,
        "market_condition": market_condition,
        "time_horizon": time_horizon,
        "industry_sector": industry_sector
    }

    with st.spinner("Menghubungi mesin komputasi..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                json=payload,
                timeout=10
            )
            
            # Jika backend mengembalikan status 4xx (misal: budget kecil tapi risk high)
            if response.status_code == 422:
                error_data = response.json()
                error_messages = [err.get("msg") for err in error_data.get("details", [])]
                st.session_state.last_error = f"❌ Validasi Ditolak: {', '.join(error_messages)}"
                st.session_state.api_result = None
            else:
                response.raise_for_status()
                st.session_state.api_result = response.json()
                st.session_state.last_error = None

        except requests.exceptions.ConnectionError:
            st.session_state.last_error = " Gagal terhubung ke backend. Pastikan Docker/FastAPI menyala di port 8000."
            st.session_state.api_result = None
        except Exception as e:
            st.session_state.last_error = f" Error tidak terduga: {str(e)}"
            st.session_state.api_result = None

# ── ERROR DISPLAY ─────────────────────────────────────────────────────
if st.session_state.last_error:
    st.error(st.session_state.last_error)

# ── RESULT DISPLAY ───────────────────────────────────────────────────
if st.session_state.api_result:
    # Mengupas wrapper respons router: {"request_id": ..., "status": ..., "data": {...}}
    raw_response = st.session_state.api_result
    req_id = raw_response.get("request_id", "N/A")
    
    # Inti output (RecommendationOutput) ada di dalam key "data"
    result_data = raw_response.get("data", {}) 

    # --- HERO SECTION: REKOMENDASI UTAMA ---
    st.header(" Hasil Rekomendasi Sistem")
    st.caption(f"Request ID: {req_id}")

    rec_scenario = result_data.get("recommended_scenario", "N/A").upper()
    confidence = result_data.get("confidence", 0.0)
    projected_revenue = result_data.get("projected_revenue", 0.0)
    reasoning = result_data.get("reasoning", "Tidak ada penjelasan.")

    # Indikator warna visual
    color_map = {"OPTIMISTIC": "🔵", "REALISTIC": "🟡", "PESSIMISTIC": "🔴"}
    rec_icon = color_map.get(rec_scenario, "⚪")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prioritas Skenario", f"{rec_icon} {rec_scenario}")
    with col2:
        st.metric("Tingkat Keyakinan", f"{confidence:.0%}")
    with col3:
        st.metric("Proyeksi Pendapatan", f"Rp {projected_revenue:,.0f}")

    st.info(
        f"**Executive Summary:** Dengan total anggaran **Rp {budget:,.0f}**, target pertumbuhan **{growth_target}x** "
        f"di sektor **{industry_sector.title()}**, dan horizon waktu **{time_horizon.title()}** — "
        f"sistem merekomendasikan eksekusi skenario **{rec_scenario}**."
    )
    
    # Memisahkan reasoning di bawahnya agar tetap rapi
    st.write(f"**Justifikasi Analitis:**\n{reasoning}")

    st.divider()

    # --- ANALISIS KOMPARATIF 3 SKENARIO ---
    st.header(" Komparasi Spread Skenario")
    
    sim_details = result_data.get("simulation_details", {})
    summary_text = sim_details.get("summary", "")
    
    if summary_text:
        st.success(summary_text)

    # Karena skema kita menggunakan atribut bernama (bukan list), kita ambil satu per satu
    opt = sim_details.get("optimistic", {})
    real = sim_details.get("realistic", {})
    pess = sim_details.get("pessimistic", {})

    if opt and real and pess:
        # Menyiapkan data untuk tabel dan grafik
        comparison_data = [
            {"Skenario": "Pessimistic", "Skor Keberhasilan": pess.get("score", 0), "Pendapatan (Rp)": pess.get("projected_revenue", 0), "Risiko Eksekusi": pess.get("risk_level", "").title()},
            {"Skenario": "Realistic", "Skor Keberhasilan": real.get("score", 0), "Pendapatan (Rp)": real.get("projected_revenue", 0), "Risiko Eksekusi": real.get("risk_level", "").title()},
            {"Skenario": "Optimistic", "Skor Keberhasilan": opt.get("score", 0), "Pendapatan (Rp)": opt.get("projected_revenue", 0), "Risiko Eksekusi": opt.get("risk_level", "").title()},
        ]
        
        df_comp = pd.DataFrame(comparison_data)

        # Layout Grafik Berdampingan
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Proyeksi Pendapatan")
            st.bar_chart(df_comp.set_index("Skenario")["Pendapatan (Rp)"], color="#2e86c1")
        with g2:
            st.subheader("Skor Keberhasilan (0-100)")
            st.bar_chart(df_comp.set_index("Skenario")["Skor Keberhasilan"], color="#28b463")

        # Tabel Detail
        st.subheader("Rincian Metrik")
        st.dataframe(df_comp, width="stretch", hide_index=True)

    # --- RAW DATA (DEBUGGING) ---
    with st.expander("Tampilkan Payload JSON Asli"):
        st.json(raw_response)