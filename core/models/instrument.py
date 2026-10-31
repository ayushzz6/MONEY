from pydantic import BaseModel


class Instrument(BaseModel):
    instrument_id: str  # e.g., NSE_RELIANCE
    symbol: str
    exchange: str
    trading_symbol: str
    angel_token: str
    sector: str
    industry: str
    active: bool
    name: str = ""
    tick_size: float = 0.05
