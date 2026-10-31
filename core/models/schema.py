from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, Boolean, BigInteger, JSON, DateTime
from datetime import datetime

Base = declarative_base()

class InstrumentDB(Base):
    __tablename__ = 'instruments'
    
    instrument_id: Mapped[str] = mapped_column(String, primary_key=True)
    symbol: Mapped[str] = mapped_column(String)
    exchange: Mapped[str] = mapped_column(String)
    token: Mapped[str] = mapped_column(String)
    sector: Mapped[str] = mapped_column(String)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

class MarketTickDB(Base):
    __tablename__ = 'market_ticks'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[int] = mapped_column(BigInteger)
    instrument_id: Mapped[str] = mapped_column(String)
    price: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)
    bid: Mapped[float] = mapped_column(Float, nullable=True)
    ask: Mapped[float] = mapped_column(Float, nullable=True)

class SignalDB(Base):
    __tablename__ = 'signals'
    
    signal_id: Mapped[str] = mapped_column(String, primary_key=True)
    timestamp: Mapped[int] = mapped_column(BigInteger)
    instrument_id: Mapped[str] = mapped_column(String)
    strategy: Mapped[str] = mapped_column(String)
    signal: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    strategy_version: Mapped[str] = mapped_column(String)

class OrderDB(Base):
    __tablename__ = 'orders'
    
    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    broker_order_id: Mapped[str] = mapped_column(String, nullable=True)
    instrument_id: Mapped[str] = mapped_column(String)
    side: Mapped[str] = mapped_column(String)
    quantity: Mapped[int] = mapped_column(Integer)
    order_type: Mapped[str] = mapped_column(String)
    price: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
