# week-14/scripts/crud_operations.py
import uuid
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import db_manager, Base
from app.models import AnalysisRequest, ScenarioResult, ModelVersion


class DSSCrudRepository:
    """Repository Pattern untuk menangani operasi database Project 3."""

    @staticmethod
    def create_analysis_with_scenarios(
        session: Session, title: str, parameters: dict, scenario_types: List[str]
    ) -> AnalysisRequest:
        """
        [CREATE] Menambahkan AnalysisRequest baru beserta data skenario anaknya sekaligus.
        Membuktikan keampuhan relasi dan cascade ORM.
        """
        # 1. Buat parent object (Transient)
        new_request = AnalysisRequest(
            title=title, parameters=parameters, status="processing"
        )

        # 2. Buat child objects dan tempelkan ke parent via properti relationship
        for s_type in scenario_types:
            scenario = ScenarioResult(
                scenario_type=s_type,
                risk_score=0.8500 if s_type == "pessimistic" else 0.3200,
                output_data={"predicted_return_pct": 12.5 if s_type == "optimistic" else 5.0}
            )
            new_request.scenarios.append(scenario)

        # 3. Cukup add parent, SQLAlchemy otomatis melakukan cascade insert ke tabel anak
        session.add(new_request)
        session.flush()  # Mengirim ke DB agar gen_random_uuid() bekerja tanpa menutup transaksi
        
        print(f"[CREATE] Berhasil menambahkan Request! ID: {new_request.id}")
        return new_request

    @staticmethod
    def get_active_requests_with_scenarios(session: Session) -> List[AnalysisRequest]:
        """
        [READ] Mengambil data request yang statusnya 'processing'
        menggunakan SQLAlchemy 2.0 Select Statement.
        """
        # Menulis query: SELECT * FROM analysis_requests WHERE status = 'processing'
        stmt = select(AnalysisRequest).where(AnalysisRequest.status == "processing")
        
        # Eksekusi query dan ambil hasilnya dalam bentuk scalar (objek murni Python)
        result = session.scalars(stmt).all()
        return result

    @staticmethod
    def update_request_status(
        session: Session, request_id: uuid.UUID, new_status: str
    ) -> Optional[AnalysisRequest]:
        """
        [UPDATE] Memperbarui status request. 
        Membuktikan otomatisasi trigger 'onupdate' pada kolom updated_at.
        """
        # 1. Cari datanya terlebih dahulu
        request = session.get(AnalysisRequest, request_id)
        
        if request:
            # 2. Ubah atribut propertinya langsung secara OOP
            request.status = new_status
            session.flush()  # Memicu pembaruan ke DB
            print(f"[UPDATE] Status Request {request_id} berhasil diubah menjadi: '{new_status}'")
            print(f"[TRIGGER CHECK] updated_at otomatis bergeser ke: {request.updated_at}")
            return request
        
        print(f"[UPDATE] Request dengan ID {request_id} tidak ditemukan.")
        return None

    @staticmethod
    def delete_analysis_request(session: Session, request_id: uuid.UUID) -> bool:
        """
        [DELETE] Menghapus request induk.
        Membuktikan keampuhan cascade="all, delete-orphan".
        """
        request = session.get(AnalysisRequest, request_id)
        if request:
            session.delete(request)
            session.flush()
            print(f"[DELETE] Request {request_id} beserta seluruh skenario anaknya DISAPU BERSIH.")
            return True
        return False


# =============================================================================
# SKRIP SIMULASI JALANNYA CRUD (MAIN EXECUTION)
# =============================================================================
def run_crud_simulation():
    print("==================================================")
    print("MULAISIMULASI OPERASI ORM CRUD PROJECT 3")
    print("==================================================\n")

    # Buka session transaksional aman via db_manager context manager
    with db_manager.session() as session:
        
        # --- 1. TEST CREATE ---
        params = {"base_capital": 50000000, "risk_appetite": "high"}
        scenarios = ["optimistic", "pessimistic"]
        inserted_req = DSSCrudRepository.create_analysis_with_scenarios(
            session=session, 
            title="Simulasi Investasi Saham Q2", 
            parameters=params, 
            scenario_types=scenarios
        )
        
        # Catat ID untuk pengujian read, update, dan delete berikutnya
        target_id = inserted_req.id

        # --- 2. TEST READ ---
        print("\n--- Menjalankan Operasi READ ---")
        active_jobs = DSSCrudRepository.get_active_requests_with_scenarios(session)
        for job in active_jobs:
            print(f"Found Request: {job.title} | Status: {job.status}")
            # Akses data anak secara instan berkat magic fungsi relationship()
            print(f"   Jumlah Skenario Anak di DB: {len(job.scenarios)}")
            for sc in job.scenarios:
                print(f"     - Tipe Skenario: {sc.scenario_type} | Risk Score: {sc.risk_score}")

        # --- 3. TEST UPDATE ---
        print("\n--- Menjalankan Operasi UPDATE (Memicu Trigger) ---")
        DSSCrudRepository.update_request_status(session, request_id=target_id, new_status="completed")

        # --- 4. TEST DELETE ---
        print("\n--- Menjalankan Operasi DELETE (Memicu Cascade) ---")
        # Jika kamu ingin melihat datanya menetap di database, bisa memberikan komentar (#) pada baris delete di bawah ini
        DSSCrudRepository.delete_analysis_request(session, request_id=target_id)

    print("\n==================================================")
    print("SIMULASI SELESAI - SEMUA TRANSAKSI DIREKAM AMAN")
    print("==================================================")


if __name__ == "__main__":
    run_crud_simulation()