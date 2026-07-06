# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base, selectinload
from typing import List, Generator

# ==========================================
# 1. Database & Models Setup (Lengkap)
# ==========================================
SQLALCHEMY_DATABASE_URL = "postgresql://dss_user:secret@localhost:5432/dss_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=20, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Tabel asosiasi untuk relasi Many-to-Many antara Post dan Tag
post_tags = Table(
    "post_tags",
    Base.metadata,
    Column("post_id", Integer, ForeignKey("posts.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

class Author(Base):
    __tablename__ = "authors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    posts = relationship("Post", back_populates="author")

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(String, nullable=False) # Kolom ini akan di-index di Alembic
    author_id = Column(Integer, ForeignKey("authors.id"), nullable=False)
    
    author = relationship("Author", back_populates="posts")
    tags = relationship("Tag", secondary=post_tags, backref="posts")

# ==========================================
# 2. Dependency Injection (Memperbaiki Bug 1)
# ==========================================
def get_db() -> Generator[Session, None, None]:
    """
    Mengelola lifecycle koneksi database secara otomatis.
    Ini menjamin koneksi akan dikembalikan ke pool (db.close()) 
    bahkan jika terjadi exception di tengah jalan.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 3. FastAPI Endpoints
# ==========================================
app = FastAPI()

@app.get("/posts")
def get_posts(db: Session = Depends(get_db)):
    # Memperbaiki Bug 2: Menggunakan selectinload untuk eager loading.
    # selectinload sangat efisien untuk relasi one-to-many (tags) 
    # karena menghindari cartesian product yang terjadi pada joinedload.
    posts = (
        db.query(Post)
        .options(
            selectinload(Post.author),
            selectinload(Post.tags)
        )
        .all()
    )

    result = []
    for post in posts:
        result.append({
            "title": post.title,
            "author": post.author.name,
            "tags": [t.name for t in post.tags],
        })

    return result

@app.get("/posts/search")
def search_posts(keyword: str, db: Session = Depends(get_db)):
    if not keyword or len(keyword.strip()) < 3:
        raise HTTPException(status_code=400, detail="Keyword terlalu pendek untuk pencarian yang efisien.")

    # Memperbaiki Bug 3: Query ini sekarang akan memanfaatkan GIN Index (pg_trgm) 
    # yang akan kita buat di migration Alembic.
    # Aku juga menambahkan limit agar hasil tidak meledak di memori.
    posts = (
        db.query(Post)
        .filter(Post.content.contains(keyword))
        .limit(100) 
        .all()
    )
    
    return [{"id": p.id, "title": p.title} for p in posts]