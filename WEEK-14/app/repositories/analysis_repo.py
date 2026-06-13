from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.analysis import AnalysisRequest, ScenarioResult

class AnalysisRepository:
    @staticmethod
    async def get_all(db: AsyncSession, skip: int, limit: int, search: str, status: str) -> list[AnalysisRequest]:
        stmt = select(AnalysisRequest)
        if search:
            stmt = stmt.where(AnalysisRequest.title.ilike(f"%{search}%"))
        if status:
            stmt = stmt.where(AnalysisRequest.status == status)
        
        stmt = stmt.offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()

    @staticmethod
    async def get_by_id(db: AsyncSession, uid: object) -> AnalysisRequest | None:
        return await db.get(AnalysisRequest, uid)

    @staticmethod
    async def create(db: AsyncSession, payload_data: dict, scenarios_data: list[dict]) -> AnalysisRequest:
        obj = AnalysisRequest(**payload_data)
        db.add(obj)
        await db.flush() # Eksekusi agar obj.id di-generate

        for sc_data in scenarios_data:
            db_scenario = ScenarioResult(request_id=obj.id, **sc_data)
            db.add(db_scenario)
            
        await db.flush()
        await db.refresh(obj)
        return obj

    @staticmethod
    async def update(db: AsyncSession, obj: AnalysisRequest, data: dict) -> AnalysisRequest:
        for key, value in data.items():
            setattr(obj, key, value)
        await db.flush()
        await db.refresh(obj)
        return obj

    @staticmethod
    async def delete(db: AsyncSession, obj: AnalysisRequest) -> None:
        await db.delete(obj)
        await db.flush()