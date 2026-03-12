"""
Database models for storing call metadata and chunks
"""

from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()


class CallTranscript(Base):
    """Stores metadata about each call transcript"""

    __tablename__ = "call_transcripts"

    id = Column(Integer, primary_key=True)
    call_id = Column(String(100), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    ingestion_date = Column(DateTime, default=datetime.utcnow)
    total_chunks = Column(Integer, default=0)

    # Relationship to chunks
    chunks = relationship(
        "CallChunk", back_populates="transcript", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<CallTranscript(call_id='{self.call_id}', filename='{self.filename}')>"


class CallChunk(Base):
    """Stores individual chunks from transcripts with their positions"""

    __tablename__ = "call_chunks"

    id = Column(Integer, primary_key=True)
    transcript_id = Column(
        Integer, ForeignKey("call_transcripts.id"), nullable=False, index=True
    )
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    timestamp_range = Column(String(50))  # e.g., "[00:00] - [01:30]"
    faiss_index = Column(Integer, nullable=False, index=True)  # Position in FAISS index

    # Relationship to transcript
    transcript = relationship("CallTranscript", back_populates="chunks")

    def __repr__(self):
        return f"<CallChunk(id={self.id}, chunk_index={self.chunk_index}, faiss_index={self.faiss_index})>"


class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, db_path: str):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)

    def get_session(self):
        """Get a new database session"""
        return self.session()

    def add_transcript(self, call_id: str, filename: str) -> CallTranscript:
        """Add a new transcript to the database"""
        session = self.get_session()
        try:
            transcript = CallTranscript(call_id=call_id, filename=filename)
            session.add(transcript)
            session.commit()
            session.refresh(transcript)
            session.expunge(transcript)
            return transcript
        finally:
            session.close()

    def add_chunk(
        self,
        transcript_id: int,
        chunk_index: int,
        text: str,
        timestamp_range: str,
        faiss_index: int,
    ) -> CallChunk:
        """Add a chunk to the database"""
        session = self.get_session()
        try:
            chunk = CallChunk(
                transcript_id=transcript_id,
                chunk_index=chunk_index,
                text=text,
                timestamp_range=timestamp_range,
                faiss_index=faiss_index,
            )
            session.add(chunk)
            session.commit()
            session.refresh(chunk)
            session.expunge(chunk)
            return chunk
        finally:
            session.close()

    def get_transcript_by_call_id(self, call_id: str):
        """Get transcript by call_id"""
        session = self.get_session()
        try:
            transcript = (
                session.query(CallTranscript).filter_by(call_id=call_id).first()
            )
            if transcript:
                session.refresh(transcript)
                session.expunge(transcript)
            return transcript
        finally:
            session.close()

    def get_all_transcripts(self):
        """Get all transcripts"""
        session = self.get_session()
        try:
            transcripts = session.query(CallTranscript).all()
            for t in transcripts:
                session.refresh(t)
                session.expunge(t)
            return transcripts
        finally:
            session.close()

    def get_chunk_by_faiss_index(self, faiss_index: int):
        """Get chunk by its FAISS index"""
        session = self.get_session()
        try:
            chunk = session.query(CallChunk).filter_by(faiss_index=faiss_index).first()
            if chunk:
                session.refresh(chunk)
                session.expunge(chunk)
            return chunk
        finally:
            session.close()

    def update_transcript_chunk_count(self, transcript_id: int, count: int):
        """Update the total chunk count for a transcript"""
        session = self.get_session()
        try:
            transcript = (
                session.query(CallTranscript).filter_by(id=transcript_id).first()
            )
            if transcript:
                transcript.total_chunks = count
                session.commit()
        finally:
            session.close()
