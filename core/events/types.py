from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class BaseEvent(BaseModel):
    event_type: str
    event_id: str
    timestamp: int

class MarketTickEvent(BaseEvent):
    event_type: str = "MARKET_TICK"
    instrument_id: str
    exchange: str
    ltp: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None

class CandleClosedEvent(BaseEvent):
    event_type: str = "CANDLE_CLOSED"
    instrument_id: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class ScannerUpdateEvent(BaseEvent):
    event_type: str = "SCANNER_UPDATE"
    top_candidates: List[str]

class RegimeChangedEvent(BaseEvent):
    event_type: str = "REGIME_CHANGED"
    old_regime: str
    new_regime: str

class SignalGeneratedEvent(BaseEvent):
    event_type: str = "SIGNAL_GENERATED"
    strategy_version: str
    instrument_id: str
    signal: str
    confidence: float
    entry_reference: Optional[float] = None
    stop_reference: Optional[float] = None
    target_reference: Optional[float] = None

class RiskDecisionEvent(BaseEvent):
    event_type: str = "RISK_DECISION"
    signal_event_id: str
    instrument_id: str
    decision: str  # ALLOW, REJECT
    reason: Optional[str] = None
    approved_quantity: Optional[int] = 0

class OrderCreatedEvent(BaseEvent):
    event_type: str = "ORDER_CREATED"
    order_id: str
    instrument_id: str
    side: str
    quantity: int
    order_type: str
    price: Optional[float] = None

class OrderFilledEvent(BaseEvent):
    event_type: str = "ORDER_FILLED"
    order_id: str
    fill_price: float
    fill_quantity: int
