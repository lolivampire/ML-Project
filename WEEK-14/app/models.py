# week-14/scripts/models.py
import uuid
from datetime import datetime
from typing import Optional, List
from decimal import Decimal

from sqlalchemy import String, Numeric, ForeignKey, DateTime, func, text, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

class AnalysisRequest(Base):
    """Parent Table: Menyimpan informasi pusat permintaan simulasi."""
    __tablename__ = "analysis_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    parameters: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, server_default="pending")
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    # Relasi dua arah ke tabel anak (Childs)
    scenarios: Mapped[List["ScenarioResult"]] = relationship(
        "ScenarioResult",
        back_populates="request",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="request",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<AnalysisRequest id={self.id} title={self.title!r} status={self.status}>"


class ScenarioResult(Base):
    """Child Table: Menyimpan output hasil komputasi skenario."""
    __tablename__ = "scenario_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_requests.id", ondelete="CASCADE"),
        nullable=False
    )
    scenario_type: Mapped[str] = mapped_column(
        String(50), nullable=False
        )
    
    # Perbaikan tipe data Python: Numeric di DB berpasangan dengan Decimal/Numeric di SQLAlchemy
    risk_score: Mapped[Optional[float]] = mapped_column(
        Numeric(precision=5, scale=4),
        nullable=True
    )
    output_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Referensi navigasi balik ke objek induk
    request: Mapped["AnalysisRequest"] = relationship(
        "AnalysisRequest",
        back_populates="scenarios"
    )
    
    efficiency_index: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
        )
    
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    def __repr__(self) -> str:
        return f"<ScenarioResult id={self.id} type={self.scenario_type} score={self.risk_score}>"


class AuditLog(Base):
    """Child Table: Jejak rekam perubahan aktivitas request."""
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analysis_requests.id", ondelete="CASCADE"),
        nullable=False
    )
    action: Mapped[str] = mapped_column(
        String(100), 
        nullable=False
        )
    changes: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True
        )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    request: Mapped["AnalysisRequest"] = relationship(
        "AnalysisRequest",
        back_populates="audit_logs"
    )

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action}>"

class ModelVersion(Base):
    """
    Model untuk melacak versi arsitektur Machine Learning.
    Menyimpan metrik performa seperti akurasi dan status keaktifan model.
    """
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()")
    )
    version_tag: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    # Menggunakan Numeric(5,4) agar presisi nilai akurasi (0.0000 - 1.0000) terjaga murni
    accuracy: Mapped[Decimal] = mapped_column(
        Numeric(precision=5, scale=4),
        nullable=True)
    # Menggunakan default=False di level Python sekaligus server_default di level DB
    is_active: Mapped[bool] = mapped_column(
        server_default=text("FALSE"),
        default=False,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    def __repr__(self) -> str:
        return f"<ModelVersion id={self.id} tag={self.version_tag} accuracy={self.accuracy}>"