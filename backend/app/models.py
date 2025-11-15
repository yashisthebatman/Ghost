import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Sessions(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_name = Column(String, unique=True, nullable=False)
    vehicle_id = Column(String, nullable=False)
    processed_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # NEW: Store weather data as a flexible JSON blob
    weather_data = Column(JSON, nullable=True, comment="Weather conditions for the session")
    
    # Relationships
    microsectors = relationship("MicroSectors", back_populates="session")
    generated_laps = relationship("GeneratedLaps", back_populates="session")
    lap_summaries = relationship("LapSummaries", back_populates="session") # New relationship

class MicroSectors(Base):
    __tablename__ = "microsectors"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    sector_type = Column(String, nullable=False, comment="e.g., 'Braking'")
    lap_number = Column(Integer)
    time_delta = Column(Float, nullable=False)
    entry_speed = Column(Float, nullable=False)
    exit_speed = Column(Float, nullable=False)
    min_speed = Column(Float, nullable=False)
    snippet_path = Column(String, unique=True, nullable=False)
    
    session = relationship("Sessions", back_populates="microsectors")

# NEW TABLE: To store official lap-by-lap summary data
class LapSummaries(Base):
    __tablename__ = "lapsummaries"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    lap_number = Column(Integer, nullable=False)
    lap_time = Column(Float)
    s1_time = Column(Float)
    s2_time = Column(Float)
    s3_time = Column(Float)
    top_speed_kph = Column(Float)
    
    session = relationship("Sessions", back_populates="lap_summaries")

class GeneratedLaps(Base):
    __tablename__ = "generatedlaps"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    mlflow_run_ids = Column(JSON)
    lap_time = Column(Float)
    ghost_lap_path = Column(String, unique=True)

    session = relationship("Sessions", back_populates="generated_laps")