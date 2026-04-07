from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")

print("DATABASE_URL loaded:", DATABASE_URL)

if DATABASE_URL is None:
    raise ValueError("DATABASE_URL is None! Check your .env file.")

engine = create_engine(
    DATABASE_URL,
    connect_args={"sslmode": "require"},
    pool_pre_ping=True

)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id                 = Column(Integer, primary_key=True, index=True)
    gender             = Column(String)
    married            = Column(String)
    dependents         = Column(String)
    education          = Column(String)
    self_employed      = Column(String)
    applicant_income   = Column(Float)
    coapplicant_income = Column(Float)
    loan_amount        = Column(Float)
    loan_amount_term   = Column(Float)
    credit_history     = Column(Float)
    property_area      = Column(String)
    result             = Column(String)
    confidence         = Column(Float)
    created_at         = Column(DateTime, default=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()