"""Local HTTP API and the dashboard, served on 127.0.0.1 only.

Every POST/DELETE must carry the header `X-MoneyV2: 1`. Browsers cannot add a custom
header to a cross-site request without a CORS preflight, which this server never
approves, so another website cannot press Approve or Flatten for you.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from apps.api_service import research

logger = logging.getLogger("API")

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
UI_FILES = ("index.html", "app.js", "style.css")


def ui_version() -> str:
    """Changes whenever a dashboard file changes: index.html asks for app.js?v=<it>, and open tabs reload on it."""
    return format(max((DASHBOARD_DIR / f).stat().st_mtime_ns for f in UI_FILES) // 1_000_000, "x")


class Revalidated(StaticFiles):
    """Static files the browser checks with the server before reusing (a 304 when unchanged), so an edit shows up."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


def create_app(engine) -> FastAPI:
    app = FastAPI(title="MONEYV2", docs_url="/api/docs", redoc_url=None)
    app.mount("/static", Revalidated(directory=str(DASHBOARD_DIR)), name="static")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method in ("POST", "DELETE", "PUT", "PATCH") and request.headers.get("x-moneyv2") != "1":
            return JSONResponse({"error": "missing X-MoneyV2 header"}, status_code=403)
        return await call_next(request)

    def _json(obj):
        return JSONResponse(json.loads(json.dumps(obj, default=str)))

    @app.get("/")
    def index():
        v = ui_version()
        html = (DASHBOARD_DIR / "index.html").read_text(encoding="utf-8")
        html = html.replace('/static/app.js"', f'/static/app.js?v={v}"').replace('/static/style.css"', f'/static/style.css?v={v}"')
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        return _json(dict(engine.snapshot(), ui_version=ui_version()))

    @app.get("/api/chart/{symbol}")
    def chart(symbol: str):
        return _json(engine.chart(symbol))

    @app.get("/api/journal")
    def journal(days: int = 30):
        return _json(engine.journal_view(days))

    @app.get("/api/backtest")
    def backtest():
        return _json(engine.backtest or {})

    @app.get("/api/config")
    def config():
        cfg = engine.settings.as_dict()
        return _json(cfg)

    def _need_ready():
        if not engine.ready:
            raise HTTPException(503, "the engine is still starting")

    @app.post("/api/trades/{trade_id}/approve")
    def approve(trade_id: str):
        _need_ready()
        return {"result": engine.approve(trade_id)}

    @app.post("/api/trades/{trade_id}/reject")
    def reject(trade_id: str):
        _need_ready()
        return {"result": engine.reject(trade_id)}

    @app.post("/api/trades/{trade_id}/exit")
    def exit_trade(trade_id: str):
        _need_ready()
        return {"result": engine.exit_trade(trade_id)}

    @app.post("/api/flatten")
    def flatten():
        _need_ready()
        return {"result": f"{engine.flatten_all()} trade(s) closing"}

    @app.post("/api/kill")
    def kill(body: dict = Body(default={})):
        engine.halt(str(body.get("reason") or "halted from the dashboard"))
        return {"result": "halted"}

    @app.post("/api/resume")
    def resume():
        return {"result": engine.resume()}

    @app.post("/api/scan")
    def scan():
        _need_ready()
        ok = engine.submit_job("prepare_day", engine.prepare_day)
        return {"result": "scan started" if ok else "a scan is already running"}

    @app.post("/api/backtest")
    def run_bt():
        _need_ready()
        ok = engine.submit_job("backtest", engine.run_backtest_job)
        return {"result": "backtest started" if ok else "a backtest is already running"}

    @app.post("/api/news")
    def news():
        _need_ready()
        engine.news.checked_at.clear()
        ok = engine.submit_job("news", engine.refresh_news)
        return {"result": "news refresh started" if ok else "already refreshing"}

    @app.post("/api/alerts")
    def add_alert(body: dict = Body(...)):
        sym = str(body.get("symbol") or "").upper().strip()
        op = body.get("op")
        try:
            price = float(body.get("price"))
        except (TypeError, ValueError):
            raise HTTPException(400, "price must be a number")
        if not sym or op not in ("above", "below") or price <= 0:
            raise HTTPException(400, "need symbol, op (above or below) and a positive price")
        if engine.universe.instruments and sym not in engine.universe.instruments and not sym.startswith("^"):
            raise HTTPException(400, f"{sym} is not in the NIFTY 500 universe")
        return engine.add_price_alert(sym, op, price, str(body.get("note") or "")[:120])

    @app.get("/api/skills")
    def skills():
        if not engine.ready:
            raise HTTPException(503, "the engine is still starting")
        return _json(engine.skills_view())

    @app.get("/api/research")
    def research_summary():
        return _json(research.summary())

    @app.get("/reports/{name}")
    def report(name: str):
        p = research.report_path(name)
        if not p:
            raise HTTPException(404, "no such report")
        return FileResponse(str(p), media_type="text/html")

    def _scope(scope: str):
        watch = [r["symbol"] for r in engine.scanner.watchlist] + [t.symbol for t in engine.book.open()]
        return watch if scope == "watchlist" else None

    @app.get("/api/intel/calendar")
    def results_calendar(scope: str = "nifty500", days: int = 14):
        import datetime as dt
        last = (dt.date.today() + dt.timedelta(days=max(1, min(days, 60)))).isoformat()
        rows = [r for r in engine.results.calendar() if r["date"] <= last]
        want = _scope(scope)
        if want is not None:
            rows = [r for r in rows if r["symbol"] in set(want)]
        elif scope == "nifty500":
            rows = [r for r in rows if r["nifty500"]]
        watch = set(_scope("watchlist"))
        return _json({"rows": [dict(r, watch=r["symbol"] in watch) for r in rows[:400]],
                      "error": engine.results.error})

    @app.get("/api/intel/results/{symbol}")
    def results_for(symbol: str):
        sym = symbol.upper().strip()
        if not sym.replace("-", "").replace("&", "").isalnum() or len(sym) > 20:
            raise HTTPException(400, "bad symbol")
        if engine.universe.instruments and sym not in engine.universe.instruments:
            raise HTTPException(400, f"{sym} is not in the NIFTY 500 universe")
        return _json(engine.results.quarterly(sym))

    @app.get("/api/intel/deals")
    def deals(scope: str = "nifty500", limit: int = 300):
        want = _scope(scope)
        rows = engine.deals.latest(max(1, min(limit, 1000)), symbols=want, nifty500_only=scope == "nifty500")
        watch = set(_scope("watchlist"))
        return _json({"rows": [dict(d, watch=d["symbol"] in watch) for d in rows], "as_of": engine.deals.as_of,
                      "error": engine.deals.error, "checked_at": engine.deals.checked_at})

    @app.get("/api/cx/board")
    def cx_board():
        return _json(engine.cx.view())

    @app.get("/api/cx/stock/{symbol}")
    def cx_stock(symbol: str):
        d = engine.cx.stock(symbol)
        if d is None:
            raise HTTPException(404, f"{symbol.upper()} is not on the crypto & XAUUSD board")
        return _json(d)

    @app.get("/api/ob/board")
    def ob_board():
        return _json(dict(engine.obboard.view(), paper=engine.board_paper("order_block"),
                          paper_first=engine.board_paper("order_block_first")))

    @app.get("/api/ob/live")
    def ob_live(sym: Optional[str] = None, after: int = 0):
        return _json(engine.obboard.live_view(sym, after))

    @app.get("/api/ob/alerts")
    def ob_alerts(after: int = 0):
        return _json(engine.obboard.alerts_view(after))

    @app.get("/api/ob/stock/{symbol}")
    def ob_stock(symbol: str):
        d = engine.obboard.stock(symbol)
        if d is None:
            raise HTTPException(404, f"{symbol.upper()} is not on the order-block board")
        return _json(d)

    @app.post("/api/intel/refresh")
    def intel_refresh():
        _need_ready()
        ok = engine.submit_job("intel", engine.refresh_intel)
        return {"result": "refreshing results calendar and deals" if ok else "already refreshing"}

    @app.delete("/api/alerts/{alert_id}")
    def del_alert(alert_id: str):
        engine.delete_price_alert(alert_id)
        return {"result": "deleted"}

    return app
