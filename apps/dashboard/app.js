/* MONEYV2 dashboard: polls the local engine and renders it. No build step. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const IST_SHIFT = 19800; // chart axis shows India time
  const UI_VERSION = new URL(document.currentScript.src).searchParams.get("v"); // the server's dashboard version
  const nf = (d) => new Intl.NumberFormat("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
  const NF0 = nf(0), NF1 = nf(1), NF2 = nf(2);
  const px = (v) => (v == null || isNaN(v) ? "–" : NF2.format(v));
  const inr = (v, d = 0) => (v == null || isNaN(v) ? "–" : (v < 0 ? "−₹" : "₹") + (d ? NF2 : NF0).format(Math.abs(v)));
  const signed = (v, d = 0) => {
    if (v == null || isNaN(v)) return "–";
    const s = v > 0 ? "+" : v < 0 ? "−" : "";
    return s + "₹" + (d ? NF2 : NF0).format(Math.abs(v));
  };
  const pnlClass = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
  const pct = (v, d = 1) => (v == null || isNaN(v) ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d) + "%");
  const cr = (v) => (v == null || isNaN(v) ? "–" : (v < 0 ? "−₹" : "₹") + (Math.abs(v) >= 100 ? NF0 : NF1).format(Math.abs(v)) + " cr");
  // bulk / block deals of one stock on one day, summed: firms that buy and sell the same day are traders, not investors
  function dealGroups(rows) {
    const g = new Map();
    for (const d of rows || []) {
      const k = d.symbol + "|" + d.date;
      if (!g.has(k)) g.set(k, { symbol: d.symbol, date: d.date, n: 0, buy: 0, sell: 0, kinds: new Set(), clients: new Map() });
      const x = g.get(k);
      x.n += 1; x.kinds.add(d.kind);
      if (d.side === "BUY") x.buy += d.value_cr || 0; else if (d.side === "SELL") x.sell += d.value_cr || 0;
      const c = x.clients.get(d.client) || new Set(); c.add(d.side); x.clients.set(d.client, c);
    }
    return [...g.values()].map((x) => {
      const both = [...x.clients.values()].filter((c) => c.has("BUY") && c.has("SELL")).length;
      return Object.assign(x, { firms: x.clients.size, roundTrip: both > 0 && both >= x.clients.size / 2 });
    }).sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : b.buy + b.sell - a.buy - a.sell));
  }
  const dealLine = (x) => `${x.n} ${[...x.kinds].join(" / ")} deal${x.n > 1 ? "s" : ""} by ${x.firms} firm${x.firms > 1 ? "s" : ""}: bought ${cr(x.buy)}, sold ${cr(x.sell)}` +
    (x.roundTrip ? ". Mostly the same firms buying and selling that day (intraday traders, not investors)." : ".");
  const fmtIST = (ts, withDate) => {
    if (!ts) return "–";
    const o = { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false };
    if (withDate) Object.assign(o, { day: "2-digit", month: "short" });
    return new Intl.DateTimeFormat("en-IN", o).format(new Date(ts * 1000));
  };
  const fmtDay = (iso) => {
    if (!iso) return "–";
    const d = new Date(iso + "T00:00:00");
    return d.toLocaleDateString("en-IN", { weekday: "short", day: "2-digit", month: "short" });
  };
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  async function api(path) {
    const r = await fetch(path, { cache: "no-store" });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.detail || j.error || path + " " + r.status);
    }
    return r.json();
  }
  async function post(path, body, method = "POST") {
    const r = await fetch(path, {
      method, headers: { "X-MoneyV2": "1", "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || j.error || r.status);
    return j;
  }
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg; t.style.display = "block";
    clearTimeout(toast._t); toast._t = setTimeout(() => (t.style.display = "none"), 4000);
  }
  const fail = (e) => toast("Did not work: " + e.message);

  // in-page dialog instead of the browser's confirm / prompt
  function ask({ title, text, ok = "OK", danger = false, input = null }) {
    return new Promise((resolve) => {
      const m = $("#modal"), inp = $("#modal-input"), okb = $("#modal-ok"), cancel = $("#modal-cancel");
      $("#modal-title").textContent = title;
      $("#modal-text").textContent = text;
      okb.textContent = ok;
      okb.className = danger ? "danger solid" : "primary";
      inp.classList.toggle("hidden", input == null);
      inp.value = input ?? "";
      m.classList.remove("hidden");
      (input != null ? inp : okb).focus();
      const done = (v) => {
        m.classList.add("hidden");
        okb.onclick = cancel.onclick = m.onclick = inp.onkeydown = document.onkeydown = null;
        resolve(v);
      };
      okb.onclick = () => done(input != null ? inp.value : true);
      cancel.onclick = () => done(null);
      m.onclick = (e) => { if (e.target === m) done(null); };
      inp.onkeydown = (e) => { if (e.key === "Enter") done(inp.value); };
      document.onkeydown = (e) => { if (e.key === "Escape") done(null); };
    });
  }

  // ------------------------------------------------------------------ state
  let S = null;
  let selected = store.get("mv2.symbol", null);
  const qs = new URLSearchParams(location.search);
  if (qs.get("symbol")) selected = qs.get("symbol").toUpperCase();
  if (["light", "dark"].includes(qs.get("theme"))) document.documentElement.setAttribute("data-theme", qs.get("theme"));
  let sigFilter = "all", ttab = "today", itab = store.get("mv2.itab", "news"), logFilter = "all";
  let journal = null, journalAt = 0, research = null, researchAt = 0, rUni = store.get("mv2.runi", null);
  let skills = null, skillsAt = 0, calRows = null, calAt = 0, dealRows = null, dealsAt = 0;
  let resultsDoc = null, resultsSym = store.get("mv2.rsym", "");
  const busy = new Set();                       // loads in flight, so a 2-second refresh does not start them again
  async function once(key, fn) {
    if (busy.has(key)) return;
    busy.add(key);
    try { await fn(); } finally { busy.delete(key); }
  }

  const PAGES = {
    home: ["Home", "Everything that matters today, at a glance."],
    live: ["Live chart", "Prices, levels and trades for one stock. Click a stock in the watchlist to see it."],
    ob: ["Order blocks", "Every F&O stock ranked by clean one-way strength each minute, with its order-block zone, levels and market alignment."],
    cx: ["Crypto & XAUUSD", "Live prices from Binance's websocket: the main coins and gold (XAUUSD), with a live chart."],
    cxob: ["Crypto & XAUUSD: order blocks", "Coins ranked by clean one-way strength today, each with its order-block zone, entry, stop-loss and target. Gold on its own."],
    cxcf: ["Crypto & XAUUSD: confirmations", "Does the rest of the evidence agree with each pair's side: futures OI, taker buying, volume, CPR, previous day, trend."],
    signals: ["Signals", "Every time a strategy's rule matched today, and what the bot did about it."],
    trades: ["Trades", "Open and closed trades, with profit or loss after all charges."],
    alerts: ["Alerts", "Price alerts you set, and a log of everything the bot did."],
    strategies: ["Strategies", "The rules the bot trades with, and which of them may trade today."],
    research: ["Research", "Five years of NSE backtests: what can make ₹20,000 a month on ₹1 lakh, and what cannot."],
    intel: ["Market intel", "News, quarterly results and big-money deals for NSE stocks."],
    skills: ["Skills", "The 11 day-trader skills and where each one runs in the bot."],
    settings: ["Settings & safety", "Mode, risk rules, data feed and background jobs."],
    help: ["Help", "How the bot works, your daily routine, and the words it uses."],
  };
  let page = PAGES[location.hash.slice(1)] ? location.hash.slice(1) : "home";

  // ------------------------------------------------------------------ theme
  const theme0 = store.get("mv2.theme", null);
  if (theme0 && !qs.get("theme")) document.documentElement.setAttribute("data-theme", theme0);
  const isDark = () => (document.documentElement.getAttribute("data-theme") ||
    (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark";
  const themeLabel = () => ($("#btn-theme").textContent = isDark() ? "Switch to light mode" : "Switch to dark mode");
  themeLabel();
  $("#btn-theme").addEventListener("click", () => {
    const next = isDark() ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    store.set("mv2.theme", next);
    themeLabel();
    applyChartTheme();
    renderPage();
  });

  // ------------------------------------------------------------------ navigation
  function setPage(p) {
    if (!PAGES[p]) p = "home";
    page = p;
    $$(".page").forEach((el) => el.classList.toggle("hidden", el.id !== "page-" + p));
    $$("#nav a").forEach((a) => (a.dataset.page === p ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
    $("#page-title").textContent = PAGES[p][0];
    $("#page-sub").textContent = PAGES[p][1];
    document.title = `${PAGES[p][0]} · MONEYV2`;
    closeMenu();
    window.scrollTo(0, 0);
    if (p === "live") { ensureChart(); if (selected) loadChart(selected, true); }
    renderPage(true);
  }
  window.addEventListener("hashchange", () => setPage(location.hash.slice(1)));
  const closeMenu = () => { $("#sidebar").classList.remove("open"); $("#scrim").classList.remove("open"); };
  $("#btn-menu").addEventListener("click", () => { $("#sidebar").classList.add("open"); $("#scrim").classList.add("open"); });
  $("#scrim").addEventListener("click", closeMenu);

  // ------------------------------------------------------------------ header actions
  $("#btn-halt").addEventListener("click", async () => {
    const reason = await ask({ title: "Pause new trades?", text: "The bot stops opening new trades. Open positions keep their stop-loss and target. You can resume any time.",
      ok: "Pause new trades", input: "paused from the dashboard" });
    if (reason === null) return;
    await post("/api/kill", { reason: reason || "paused from the dashboard" }).catch(fail);
    refresh();
  });
  $("#btn-resume").addEventListener("click", async () => {
    const r = await post("/api/resume").catch(fail);
    if (r) toast(r.result === "resumed" ? "Trading resumed: new trades allowed again" : r.result);
    refresh();
  });
  $("#btn-flatten").addEventListener("click", async () => {
    const n = openTrades().length;
    const ok = await ask({ title: "Exit all positions?", text: n ? `This closes all ${n} open position(s) at market price right now.` : "There are no open positions right now.",
      ok: "Exit all now", danger: true });
    if (!ok) return;
    const r = await post("/api/flatten").catch(fail);
    if (r) toast(r.result);
    refresh();
  });

  // one click handler for every action button on the page
  document.addEventListener("click", async (e) => {
    const symEl = e.target.closest("[data-sym]");
    const b = e.target.closest("button");
    if (b && b.dataset.approve) {
      b.disabled = true;
      const r = await post(`/api/trades/${encodeURIComponent(b.dataset.approve)}/approve`).catch(fail);
      if (r) toast("Approve: " + r.result);
      refresh();
    } else if (b && b.dataset.reject) {
      b.disabled = true;
      await post(`/api/trades/${encodeURIComponent(b.dataset.reject)}/reject`).catch(fail);
      refresh();
    } else if (b && b.dataset.exit) {
      const ok = await ask({ title: `Exit ${b.dataset.exitSym || "this position"}?`, text: "The position is closed at market price now.", ok: "Exit now", danger: true });
      if (!ok) return;
      b.disabled = true;
      const r = await post(`/api/trades/${encodeURIComponent(b.dataset.exit)}/exit`).catch(fail);
      if (r) toast("Exit: " + r.result);
      refresh();
    } else if (b && b.dataset.delalert) {
      await post(`/api/alerts/${encodeURIComponent(b.dataset.delalert)}`, null, "DELETE").catch(fail);
      refresh();
    } else if (b && b.dataset.job) {
      b.disabled = true;
      const r = await post(`/api/${b.dataset.job}`).catch(fail);
      if (r) toast(r.result);
      if (b.dataset.job === "intel/refresh") { calAt = dealsAt = 0; }
      setTimeout(() => { b.disabled = false; refresh(); }, 800);
    } else if (b && b.dataset.sig) {
      sigFilter = b.dataset.sig; renderSignals();
    } else if (b && b.dataset.ttab) {
      ttab = b.dataset.ttab; renderTrades(true);
    } else if (b && b.dataset.itab) {
      itab = b.dataset.itab; store.set("mv2.itab", itab); renderIntel(true);
    } else if (b && b.dataset.obtab) {
      obTab = b.dataset.obtab; store.set("mv2.obtab", obTab); renderOB(false); drawOB();
    } else if (b && b.dataset.obalerts) {
      toggleAlerts();
    } else if (e.target.closest("[data-cx]") && !b) {
      e.preventDefault();
      cxSelect(e.target.closest("[data-cx]").dataset.cx);
    } else if (e.target.closest("[data-ob-go]")) {
      obSel = e.target.closest("[data-ob-go]").dataset.obGo; store.set("mv2.obsym", obSel); location.hash = "#ob";
    } else if (e.target.closest("[data-ob]") && !b) {
      selectOB(e.target.closest("[data-ob]").dataset.ob);
    } else if (b && b.dataset.runi) {
      rUni = b.dataset.runi; store.set("mv2.runi", rUni); renderResearch.force = true; renderResearch();
    } else if (b && b.dataset.results) {
      e.preventDefault();
      location.hash = "#intel"; itab = "results"; store.set("mv2.itab", itab);
      lookupResults(b.dataset.results);
    } else if (symEl && !b) {
      selectSymbol(symEl.dataset.sym, true);
    }
  });
  document.addEventListener("keydown", (e) => {
    const cxEl = e.target.closest && e.target.closest("[data-cx]");
    if (cxEl && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); cxSelect(cxEl.dataset.cx); return; }
    const obEl = e.target.closest && e.target.closest("[data-ob]");
    if (obEl && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); selectOB(obEl.dataset.ob); return; }
    const el = e.target.closest && e.target.closest("[data-sym]");
    if (el && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); selectSymbol(el.dataset.sym, true); }
  });

  // ------------------------------------------------------------------ polling
  async function refresh() {
    try {
      S = await api("/api/state");
      // the dashboard files changed on the server: load them (never while a LIVE trade waits for Approve)
      if (UI_VERSION && S.ui_version && S.ui_version !== UI_VERSION && !pendingTrades().length) { location.reload(); return; }
      render();
    } catch (e) {
      $("#st-market").innerHTML = `<span class="dot bad"></span><span>Bot offline: run ./run.sh</span>`;
    }
  }
  setInterval(refresh, 2000);
  setInterval(() => { if (selected && page === "live") loadChart(selected, false); }, 5000);

  function render() {
    renderShell();
    renderPage();
    obAlertsPoll();
    if (!selected) {
      const open = openTrades()[0];
      const first = open ? open.symbol : (S.watchlist[0] || {}).symbol;
      if (first) selectSymbol(first, false);
    }
  }
  function renderPage(entering) {
    if (!S) return;
    ({
      home: renderHome, live: renderLive, ob: () => renderOB(entering), cx: () => renderCX(entering), cxob: () => renderCX(entering), cxcf: () => renderCX(entering), signals: renderSignals, trades: renderTrades, alerts: renderAlerts,
      strategies: renderStrategies, research: renderResearch, intel: () => renderIntel(entering), skills: renderSkills,
      settings: renderSettings, help: renderHelp,
    })[page]();
  }

  const ACTIVE = ["PENDING_APPROVAL", "APPROVED", "ENTRY_PLACED", "OPEN", "EXITING"];
  const openTrades = () => (S ? (S.trades || []).filter((t) => ACTIVE.includes(t.status)) : []);
  const pendingTrades = () => (S ? (S.trades || []).filter((t) => t.status === "PENDING_APPROVAL") : []);

  // ------------------------------------------------------------------ shell: status pills, mode, banners, nav counts
  function marketText(c) {
    if (c.state === "OPEN") {
      const h = Math.floor(c.minutes_to_close / 60), mm = c.minutes_to_close % 60;
      return [`Market open · closes in ${h}h ${mm}m`, "ok"];
    }
    if (c.state === "PRE_OPEN" && c.minutes_to_open <= 180) return [`Market opens in ${c.minutes_to_open} min`, "warn"];
    const nxt = c.next_open ? c.next_open.replace(/^(\w+) \d{4}-(\d\d)-(\d\d) (\d\d:\d\d)$/, "$1 $3/$2 $4") : "?";
    const why = { WEEKEND: "Weekend", HOLIDAY: "Market holiday", CLOSED: "Market closed", PRE_OPEN: "Market closed" }[c.state] || "Market closed";
    return [`${why} · opens ${nxt}`, ""];
  }
  function renderShell() {
    const c = S.clock || {};
    const [mt, md] = marketText(c);
    $("#st-market").innerHTML = `<span class="dot ${md}"></span><span>${esc(mt)} · ${esc((c.now || "").slice(11, 16))} IST</span>`;
    const f = S.feed || {};
    let ft, fd = "";
    if (f.mode === "angel") {
      if (f.angel_connected) { ft = "Angel One live prices"; fd = "ok"; }
      else if (f.standby) { ft = "Angel One: reconnects before the open"; fd = ""; }
      else { ft = "Angel One reconnecting"; fd = "bad"; }
    }
    else {
      const age = f.last_quotes_age_s;
      ft = "Yahoo prices (delayed)";
      if (c.state === "OPEN") fd = age == null ? "" : age > 180 ? "bad" : age > 60 ? "warn" : "ok";
    }
    $("#st-data").innerHTML = `<span class="dot ${fd}"></span><span>${esc(ft)}</span>`;
    const g = (S.kill || {}).global;
    $("#st-safety").innerHTML = g ? `<span class="dot bad"></span><span>New trades paused</span>` : `<span class="dot ok"></span><span>New trades allowed</span>`;
    $("#st-safety").title = g ? `Paused: ${g.reason}` : "The bot may open new trades within your risk rules";
    $("#btn-halt").classList.toggle("hidden", !!g);
    $("#btn-resume").classList.toggle("hidden", !g);
    const live = S.mode === "LIVE";
    const mb = $("#mode-box");
    mb.className = "mode-box" + (live ? " live" : "");
    mb.innerHTML = live ? `<b>LIVE · real money</b>Every new trade waits for your Approve.` : `<b>Paper mode</b>Practice money. No real orders are sent.`;
    const fatal = $("#banner-fatal");
    fatal.classList.toggle("hidden", !S.fatal);
    fatal.textContent = S.fatal || "";
    // approvals: shown on every page
    const pend = pendingTrades();
    const ab = $("#banner-approve");
    ab.classList.toggle("hidden", !pend.length);
    if (pend.length) {
      const timeout = (S.rules || {}).approval_timeout_seconds || 120;
      ab.innerHTML = `<h3>${pend.length} trade${pend.length > 1 ? "s" : ""} waiting for your approval (real money)</h3>` + pend.map((t) => {
        const left = Math.max(0, Math.round(timeout - (Date.now() / 1000 - t.created_at)));
        return `<div class="approval-row"><div class="desc"><b>${t.side === "buy" ? "BUY" : "SELL"} ${t.qty} ${esc(t.symbol)}</b> near ${px(t.plan.entry)} ·
          stop-loss ${px(t.stop)} · target ${px(t.target)} · <b>most you can lose ${inr(t.max_loss)}</b> · ${esc(t.strategy)}
          <span class="muted">· expires in ${left}s</span></div>
          <button class="primary" data-approve="${esc(t.id)}">Approve</button><button data-reject="${esc(t.id)}">Reject</button></div>`;
      }).join("");
    }
    const setCount = (id, n, hot) => { const el = $(id); el.textContent = n ? String(n) : ""; el.classList.toggle("hot", !!hot && n > 0); };
    setCount("#nc-home", pend.length, true);
    setCount("#nc-signals", (S.signals || []).length);
    setCount("#nc-trades", openTrades().length);
    setCount("#nc-intel", ((S.intel || {}).results_soon || []).length);
    setCount("#nc-ob", (S.ob_alerts || {}).active || 0, true);           // order-block heads-up right now
  }

  // ------------------------------------------------------------------ HOME
  function heroSentence() {
    const c = S.clock || {}, r = S.rules || {};
    const wl = (S.watchlist || []).length;
    const trading = (S.strategies || []).filter((s) => s.trading);
    const parts = [];
    parts.push(S.mode === "LIVE"
      ? `The bot is running in <b>LIVE mode</b>: real money through Angel One, and every new trade waits for your Approve.`
      : `The bot is running in <b>Paper mode</b>: it trades by itself with practice money, so nothing here is real.`);
    if (c.state === "OPEN") {
      const h = Math.floor(c.minutes_to_close / 60), mm = c.minutes_to_close % 60;
      parts.push(`The market is <b>open</b> and closes in ${h}h ${mm}m.`);
    } else if (c.state === "PRE_OPEN" && c.minutes_to_open <= 180) {
      parts.push(`The market opens in <b>${c.minutes_to_open} minutes</b>.`);
    } else {
      parts.push(`The market is <b>closed</b>. The next session starts <b>${esc(marketText(c)[0].replace(/^.* · opens /, ""))}</b>.`);
    }
    if (wl) parts.push(`It is watching <b>${wl} stocks</b>${S.trade_date ? ` for ${esc(fmtDay(S.trade_date))}` : ""}, with <b>${trading.length} of ${(S.strategies || []).length} strategies</b> allowed to trade${trading.length ? ` (${trading.map((s) => esc(s.label)).join(", ")})` : ""}.`);
    else parts.push(`The morning scan is ${((S.jobs || {}).prepare_day || {}).state === "running" ? "running now" : "getting ready"}: the watchlist appears in a few minutes.`);
    const g = (S.kill || {}).global;
    if (g) parts.push(`<b>New trades are paused</b> (${esc(g.reason)}).`);
    else parts.push(`New trades are allowed between ${esc(r.no_entries_before)} and ${esc(r.no_entries_after)}; everything is closed by ${esc(r.flat_by)}.`);
    return parts.join(" ");
  }
  function timeline() {
    const c = S.clock || {}, r = S.rules || {};
    const steps = [["Morning scan", "before 09:15"], ["Market opens", "09:15"], ["New trades allowed", r.no_entries_before],
      ["Last new trade", r.no_entries_after], ["Everything closed", r.flat_by], ["Market closes", "15:30"]];
    const hm = (c.now || "").slice(11, 16);
    let cur = -1;
    if (c.state === "OPEN") { cur = 1; steps.forEach(([, t], i) => { if (i >= 1 && /^\d\d:\d\d$/.test(t) && hm >= t) cur = i; }); }
    else if (c.state === "PRE_OPEN") cur = 0;
    else if (c.state === "CLOSED") cur = steps.length;
    const label = c.state === "WEEKEND" || c.state === "HOLIDAY" || (c.state === "PRE_OPEN" && c.minutes_to_open > 180)
      ? "Next session" : c.state === "CLOSED" ? "Today's session (finished)" : "Today's session";
    return `<div class="small muted">${label}</div><div class="timeline">${steps.map(([l, t], i) =>
      `<div class="step ${i < cur ? "done" : i === cur ? "now" : ""}"><b>${esc(t)}</b>${esc(l)}</div>`).join("")}</div>`;
  }
  function renderHome() {
    $("#home-hero").innerHTML = `<div class="sentence">${heroSentence()}</div>${timeline()}`;
    // needs attention
    const items = [];
    const g = (S.kill || {}).global;
    if (S.fatal) items.push(["bad", S.fatal]);
    if (g) items.push(["warn", `New trades are paused: ${g.reason}. Press "Resume trading" when you are ready.`]);
    pendingTrades().forEach((t) => items.push(["warn", `${t.symbol}: a ${t.side === "buy" ? "buy" : "sell"} of ${t.qty} shares is waiting for your approval (yellow bar above).`]));
    (S.intel || {}).results_soon?.forEach((r) => items.push(["", `${r.symbol} announces results on ${fmtDay(r.date)}. Its price can jump that day.`]));
    const hourAgo = Date.now() / 1000 - 3600;
    (S.alerts || []).filter((a) => a.kind === "error" && a.ts > hourAgo).slice(0, 3).forEach((a) => items.push(["bad", a.text]));
    $("#home-attention").innerHTML = items.length ? `<div class="card"><div class="card-head"><h2>Needs your attention</h2></div><ul class="feed">${items.map(([cls, t]) =>
      `<li><span class="chip ${cls}">${cls === "bad" ? "Problem" : cls === "warn" ? "Action" : "Heads-up"}</span> ${esc(t)}</li>`).join("")}</ul></div>` : "";
    // tiles
    const a = S.account || {};
    const used = a.risk_used_pct || 0;
    $("#home-tiles").innerHTML = `
      <div class="tile"><div class="label">Profit / loss today (after charges)</div><div class="value ${pnlClass(a.net)}">${signed(a.net)}</div>
        <div class="sub">closed ${signed(a.realized)} · open ${signed(a.unrealized)}</div></div>
      <div class="tile"><div class="label">Daily loss limit used</div><div class="value">${inr(a.risk_used)} <span class="muted small">of ${inr(a.daily_limit)}</span></div>
        <div class="meter ${used >= 80 ? "hot" : ""}" role="meter" aria-valuenow="${used}" aria-valuemin="0" aria-valuemax="100"><span style="width:${Math.min(100, used)}%"></span></div>
        <div class="sub">losses so far + what open trades lose if stopped</div></div>
      <div class="tile"><div class="label">Trades today</div><div class="value">${a.entries_today ?? 0} <span class="muted small">of ${a.max_trades ?? "–"} allowed</span></div>
        <div class="sub">${(S.signals || []).length} signals seen</div></div>
      <div class="tile"><div class="label">Open positions</div><div class="value">${a.open_positions ?? 0} <span class="muted small">of ${a.max_open ?? "–"}</span></div>
        <div class="sub">capital ${inr(a.capital)} · max loss a trade ${inr(a.max_loss_per_trade)}</div></div>`;
    // open positions
    const open = openTrades();
    $("#home-positions").innerHTML = open.length ? `<div class="table-wrap"><table><thead><tr><th>Stock</th><th>Side</th><th class="r">Qty</th><th class="r">Entry</th><th class="r">Now</th><th class="r">P&amp;L</th><th></th></tr></thead><tbody>${open.map((t) =>
      `<tr class="clickable" data-sym="${esc(t.symbol)}"><td><b>${esc(t.symbol)}</b><span class="name">${esc(t.strategy)}</span></td><td>${t.side === "buy" ? "Long" : "Short"}</td><td class="r">${t.qty}</td>
        <td class="r">${px(t.entry_price || t.plan.entry)}</td><td class="r">${px(t.ltp)}</td><td class="r ${pnlClass(t.unrealized)}">${signed(t.unrealized, 2)}</td>
        <td>${t.status === "OPEN" ? `<button class="small" data-exit="${esc(t.id)}" data-exit-sym="${esc(t.symbol)}">Exit</button>` : `<span class="chip">${esc(statusText(t.status))}</span>`}</td></tr>`).join("")}</tbody></table></div>`
      : `<p class="empty">No open positions.</p>`;
    // signals
    const sigs = (S.signals || []).slice(0, 6);
    $("#home-signals").innerHTML = sigs.length ? sigs.map(signalItem).join("") : `<li class="empty">${(S.clock || {}).state === "OPEN" ? "No setup has triggered yet today." : "Signals appear here while the market is open (09:15 to 15:30)."}</li>`;
    // coming up
    const intel = S.intel || {};
    const up = [...(intel.results_soon || []).map((r) => `<li><div class="row1"><span class="chip blue">Results</span><span class="sym" data-sym="${esc(r.symbol)}" tabindex="0">${esc(r.symbol)}</span><span class="muted small">${esc(fmtDay(r.date))}</span>
        <button class="link small" data-results="${esc(r.symbol)}">Past results →</button></div></li>`),
      ...dealGroups(intel.watch_deals).slice(0, 4).map((x) => `<li><div class="row1"><span class="chip">Big deals</span><span class="sym" data-sym="${esc(x.symbol)}" tabindex="0">${esc(x.symbol)}</span><span class="muted small">${esc(fmtDay(x.date))}</span></div>
        <div class="why">${esc(dealLine(x))}</div></li>`)];
    $("#home-upcoming").innerHTML = up.length ? `<ul class="feed">${up.join("")}</ul>` : `<p class="empty">No results due in the next 3 days and no big deals in your watchlist stocks.</p>`;
    // activity
    const act = (S.alerts || []).slice(0, 8);
    $("#home-activity").innerHTML = act.length ? act.map(alertItem).join("") : `<li class="empty">Nothing yet.</li>`;
    renderHomeOB();
  }

  const STATUS = { PENDING_APPROVAL: "waiting for approval", APPROVED: "approved", ENTRY_PLACED: "entry order placed", OPEN: "open",
    EXITING: "closing", CLOSED: "closed", REJECTED: "rejected", EXPIRED: "expired", CANCELLED: "cancelled", ERROR: "error" };
  const statusText = (s) => STATUS[s] || String(s || "").toLowerCase().replace(/_/g, " ");

  // ------------------------------------------------------------------ signals
  const DECISIONS = {
    TRADED: ["good", "Traded"], AWAITING_APPROVAL: ["warn", "Approve?"], BLOCKED: ["bad", "Blocked"],
    SKIPPED: ["", "Skipped"], BENCHED: ["", "Benched"], REFUSED: ["", "Refused"], STALE: ["", "Stale data"],
  };
  const decisionChip = (d) => { const [cls, label] = DECISIONS[d] || ["", d]; return `<span class="chip ${cls}">${esc(label)}</span>`; };
  function signalItem(s) {
    const p = s.plan, tm = s.tm, mc = tm && tm.mc, h = tm && tm.history;
    const facts = [];
    if (p) facts.push(`entry ${px(p.entry)}`, `stop-loss ${px(p.stop)}`, `target ${px(p.target)}`, `aims to make ${p.reward_risk}× its risk`);
    else facts.push(`price ${px(s.entry)}`, `level ${px(s.level)}`);
    if (s.size && s.size.qty) facts.push(`${s.size.qty} shares · most it can lose ${inr(s.size.max_loss)}`);
    const tmLine = mc ? `<div class="facts"><span>Time Machine: reached target in ${(100 * mc.p_target).toFixed(0)}% of replays, stop-loss in ${(100 * mc.p_stop).toFixed(0)}%</span>
      <span>past times this fired: ${h.triggered}${h.avg_R != null ? ` (avg ${h.avg_R >= 0 ? "+" : ""}${h.avg_R}R)` : ""}</span></div>${tm.verdict ? `<div class="verdict">${esc(tm.verdict)}</div>` : ""}` : "";
    return `<li><div class="row1"><span class="num muted">${esc(s.time)}</span>
        <span class="sym" data-sym="${esc(s.symbol)}" tabindex="0">${esc(s.symbol)}</span>
        <span class="chip">${s.side === "buy" ? "Buy" : "Sell short"}</span><span class="ink2 small">${esc(s.label || s.strategy)}</span>
        ${decisionChip(s.decision)}</div>
      <div class="why">${esc(s.why)}</div><div class="facts">${facts.map((f) => `<span>${f}</span>`).join("")}</div>${tmLine}</li>`;
  }
  function renderSignals() {
    const rows = S.signals || [];
    const groups = [["all", "All"], ["TRADED", "Traded"], ["AWAITING_APPROVAL", "Waiting for you"], ["BLOCKED", "Blocked"], ["other", "Benched / skipped"]];
    const count = (k) => k === "all" ? rows.length : k === "other" ? rows.filter((s) => !["TRADED", "AWAITING_APPROVAL", "BLOCKED"].includes(s.decision)).length : rows.filter((s) => s.decision === k).length;
    $("#sig-filters").innerHTML = groups.map(([k, l]) => `<button data-sig="${k}" aria-pressed="${sigFilter === k}">${l} (${count(k)})</button>`).join("");
    const shown = rows.filter((s) => sigFilter === "all" || (sigFilter === "other" ? !["TRADED", "AWAITING_APPROVAL", "BLOCKED"].includes(s.decision) : s.decision === sigFilter));
    $("#sig-list").innerHTML = shown.length ? shown.slice(0, 150).map(signalItem).join("")
      : `<li class="empty">${rows.length ? "No signals of this kind today." : (S.clock || {}).state === "OPEN" ? "No setup has triggered yet. Every trigger shows here with what the bot did and why." : "Signals appear here while the market is open (09:15 to 15:30 IST)."}</li>`;
  }

  // ------------------------------------------------------------------ trades
  function tradeRows(rows) {
    return `<table><thead><tr><th>Time</th><th>Stock</th><th>Strategy</th><th>Side</th><th class="r">Qty</th><th class="r">Entry</th><th class="r">Now / exit</th>
      <th class="r">Stop-loss</th><th class="r">Target</th><th class="r">P&amp;L</th><th class="r">R</th><th>Status</th><th></th></tr></thead><tbody>` +
      rows.map((t) => {
        const pnl = t.status === "CLOSED" ? t.net_pnl : t.unrealized;
        const now = t.status === "CLOSED" ? t.exit_price : t.ltp;
        const st = t.status === "CLOSED" ? `closed: ${t.exit_reason}` : statusText(t.status);
        let act = "";
        if (t.status === "OPEN") act = `<button class="small" data-exit="${esc(t.id)}" data-exit-sym="${esc(t.symbol)}">Exit</button>`;
        if (t.status === "PENDING_APPROVAL") act = `<button class="small primary" data-approve="${esc(t.id)}">Approve</button> <button class="small" data-reject="${esc(t.id)}">Reject</button>`;
        return `<tr class="clickable ${t.symbol === selected ? "selected" : ""}" data-sym="${esc(t.symbol)}">
          <td class="num">${fmtIST(t.opened_at || t.created_at)}</td><td><b>${esc(t.symbol)}</b></td><td class="small">${esc(t.strategy)}</td>
          <td>${t.side === "buy" ? "Long" : "Short"}</td><td class="r">${t.qty}</td><td class="r">${px(t.entry_price || t.plan.entry)}</td><td class="r">${px(now)}</td>
          <td class="r">${px(t.stop)}</td><td class="r">${px(t.target)}</td><td class="r ${pnlClass(pnl)}">${signed(pnl, 2)}</td>
          <td class="r">${t.R != null ? (t.R >= 0 ? "+" : "") + t.R.toFixed(2) : "–"}</td><td class="small">${esc(st)}</td><td>${act}</td></tr>`;
      }).join("") + "</tbody></table>";
  }
  function renderTrades(entering) {
    $$("#trade-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.ttab === ttab)));
    $("#trades-today").classList.toggle("hidden", ttab !== "today");
    $("#trades-history").classList.toggle("hidden", ttab !== "history");
    if (ttab === "today") {
      const rows = S.trades || [];
      $("#trades-note").textContent = S.mode === "LIVE" ? "Angel One · intraday (MIS)" : "paper broker · simulated fills";
      $("#trades-table").innerHTML = rows.length ? tradeRows(rows) : `<p class="empty">No trades today. Trades appear here as soon as the bot opens one.</p>`;
    } else if (entering || !journal || Date.now() - journalAt > 60000) {
      once("journal", async () => {
        journalAt = Date.now();
        try { journal = await api("/api/journal?days=90"); } catch (e) { $("#trades-history").innerHTML = `<div class="card"><p class="empty">Journal unavailable.</p></div>`; return; }
        renderJournal();
      });
    }
  }
  function renderJournal() {
    const el = $("#trades-history");
    const j = journal, days = j.days || [];
    const tot = days.reduce((s, d) => s + d.net, 0), n = days.reduce((s, d) => s + d.trades, 0), w = days.reduce((s, d) => s + d.wins, 0);
    el.innerHTML = `<div class="tiles">
        <div class="tile"><div class="label">Net, last 90 days</div><div class="value ${pnlClass(tot)}">${signed(tot)}</div><div class="sub">paper and live together</div></div>
        <div class="tile"><div class="label">Closed trades</div><div class="value">${n}</div><div class="sub">${n ? Math.round(100 * w / n) + "% winners" : "–"}</div></div>
        <div class="tile"><div class="label">Trading days</div><div class="value">${days.length}</div><div class="sub">${days.filter((d) => d.net > 0).length} winning days</div></div></div>
      <div class="card"><h3>Profit / loss calendar</h3><div id="calheat"></div></div>
      <div class="grid-2">
        <div class="card"><h3>Net profit / loss by day</h3><div id="daybars"></div></div>
        <div class="card"><h3>By strategy</h3>${(j.strategies || []).length ? `<div class="table-wrap"><table><thead><tr><th>Strategy</th><th class="r">Trades</th><th class="r">Winners</th><th class="r">Net</th><th class="r">Avg R</th></tr></thead><tbody>${j.strategies.map((s) => `<tr><td>${esc(s.strategy)}</td><td class="r">${s.trades}</td><td class="r">${(100 * s.wins / s.trades).toFixed(0)}%</td><td class="r ${pnlClass(s.net)}">${signed(s.net)}</td><td class="r">${(s.R / s.trades).toFixed(2)}</td></tr>`).join("")}</tbody></table></div>` : `<p class="empty">No closed trades yet.</p>`}</div></div>
      <div class="card flush"><div class="card-head"><h2>Closed trades</h2></div><div class="table-wrap scroll" style="max-height:420px">${(j.trades || []).length ? `<table><thead><tr><th>Date</th><th>Stock</th><th>Strategy</th><th>Mode</th><th class="r">Qty</th><th class="r">Entry</th><th class="r">Exit</th><th>Why it closed</th><th class="r">Net</th><th class="r">R</th></tr></thead><tbody>${j.trades.slice().reverse().map((t) => `<tr><td class="num">${esc(t.date)}</td><td>${esc(t.symbol)}</td><td class="small">${esc(t.strategy)}</td><td class="small">${esc(t.mode)}</td><td class="r">${t.qty}</td><td class="r">${px(t.entry_price)}</td><td class="r">${px(t.exit_price)}</td><td class="small">${esc(t.exit_reason)}</td><td class="r ${pnlClass(t.net_pnl)}">${signed(t.net_pnl, 2)}</td><td class="r">${t.R != null ? t.R.toFixed(2) : "–"}</td></tr>`).join("")}</tbody></table>` : `<p class="empty">No closed trades in the last 90 days.</p>`}</div></div>`;
    calHeat($("#calheat"), days);
    barChart($("#daybars"), days.map((d) => ({ x: d.date, y: d.net, n: d.trades, w: d.wins })),
      (p) => `${p.x}<br>net ${signed(p.y)}<br>${p.n} trades, ${p.w} winners`);
  }

  // ------------------------------------------------------------------ alerts
  const KINDS = {
    error: ["bad", "Problem"], warn: ["warn", "Warning"], approval: ["warn", "Approve"], price: ["warn", "Price alert"],
    fill: ["good", "Filled"], closed: ["", "Closed"], order: ["", "Order"], news: ["", "News"], bell: ["", "Bell"],
    flat: ["", "Flat-by"], expired: ["", "Expired"], blocked: ["bad", "Blocked"], info: ["", "Info"],
    results: ["blue", "Results"], deal: ["", "Big deal"],
  };
  function alertItem(a) {
    const [cls, label] = KINDS[a.kind] || ["", a.kind];
    return `<li><div class="row1"><span class="num muted">${fmtIST(a.ts)}</span><span class="chip ${cls}">${esc(label)}</span>
      ${a.symbol ? `<span class="sym" data-sym="${esc(a.symbol)}" tabindex="0">${esc(a.symbol)}</span>` : ""}</div><div class="why">${esc(a.text)}</div></li>`;
  }
  function renderAlerts() {
    const pa = S.price_alerts || [];
    $("#palerts").innerHTML = pa.length ? `<table><thead><tr><th>Stock</th><th>Rings when price</th><th class="r">Price</th><th>Note</th><th>Status</th><th></th></tr></thead><tbody>` +
      pa.map((a) => `<tr><td><b>${esc(a.symbol)}</b></td><td>${a.op === "above" ? "rises to or above" : "falls to or below"}</td><td class="r">${px(a.price)}</td><td>${esc(a.note)}</td>
        <td class="small">${a.fired_at ? "rang " + fmtIST(a.fired_at, true) : "watching"}</td><td><button class="small" data-delalert="${esc(a.id)}">Delete</button></td></tr>`).join("") + "</tbody></table>"
      : `<p class="empty">No price alerts yet.</p>`;
    const rows = S.alerts || [];
    const kinds = [...new Set(rows.map((a) => a.kind))];
    const sel = $("#log-filter");
    const want = ["all", ...kinds].join(",");
    if (sel.dataset.k !== want) {
      sel.dataset.k = want;
      sel.innerHTML = `<option value="all">Everything</option>` + kinds.map((k) => `<option value="${esc(k)}">${esc((KINDS[k] || ["", k])[1])}</option>`).join("");
      sel.value = kinds.includes(logFilter) ? logFilter : "all";
    }
    const shown = rows.filter((a) => logFilter === "all" || a.kind === logFilter);
    $("#log-list").innerHTML = shown.length ? shown.map(alertItem).join("") : `<li class="empty">Nothing yet.</li>`;
  }
  $("#log-filter").addEventListener("change", (e) => { logFilter = e.target.value; renderAlerts(); });
  $("#alert-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = new FormData(e.target);
    const r = await post("/api/alerts", { symbol: f.get("symbol"), op: f.get("op"), price: f.get("price"), note: f.get("note") }).catch(fail);
    if (r) { e.target.reset(); toast(`Alert set: ${r.symbol} ${r.op === "above" ? "at or above" : "at or below"} ${px(r.price)}`); refresh(); }
  });

  // ------------------------------------------------------------------ strategies
  const stopText = (s) => {
    s = String(s || "");
    if (s === "or_mid") return "the middle of the opening range";
    if (s === "or_low" || s === "or") return "the other side of the opening range";
    if (s === "zone") return "the zone's invalidation level, just past the order block";
    if (s.startsWith("swing")) { const n = s.split(":")[1]; return n ? `just past the lowest point of the last ${n} candles` : "just past the last swing low"; }
    if (s.startsWith("atr")) return `${s.split(":")[1] || s.slice(3)} × the daily range (ATR) away`;
    if (s.startsWith("pct")) return `${s.split(":")[1] || s.slice(3)}% away`;
    return s || "set by the strategy";
  };
  const HOW = {
    orb: (p) => [`Marks the high and low of the first ${p.or_minutes} minutes (the "opening range").`,
      `Buys when a 5-minute candle closes above the range high${p.short ? ", or sells short when one closes below the low" : ""}.`,
      `Only when the range is ${p.min_or_pct}–${p.max_or_pct}% of the price${p.min_or_rvol ? ` and volume is at least ${p.min_or_rvol}× normal` : ""}.`],
    level_breakout: (p) => [`Watches stocks within ${p.near_pct}% of their 20-day high.`,
      `Buys when the price breaks above that high, unless the stock opened more than ${p.max_gap_pct}% up (a gap often fades).`],
    vwap_reclaim: (p) => [`For stocks in an uptrend that dipped below VWAP (today's average price, weighted by volume).`,
      `Buys when the price climbs back above VWAP, not before ${p.not_before}.`],
    order_block: (p) => [`Takes the Order blocks page's picks${p.min_aligned ? ` with ${p.min_aligned} of 4 checks aligned` : ""}, while price has not yet come back to the zone.`,
      `Enters the moment the live price comes back to the zone's entry level Z (bearish zones only when short selling is on). One try per stock a day.`,
      ...(p.paper_only ? [`Paper only: it never trades real money. Its 2-year backtest lost money; this is a forward test.`] : [])],
    order_block_first: (p) => [`Takes the same picks${p.min_aligned ? ` with ${p.min_aligned} of 4 checks aligned` : ""}, but does not wait for the retest.`,
      `Enters at the live price in the ${(p.window_bars || 1) * 5} minutes after the zone is given, before the price comes back to Z. Stop: the zone's invalidation; target: the levels as given when the zone appeared. One try per stock a day.`,
      `The stop is further away than at Z, so the risk rules' minimum reward:risk refuses the entries whose target is too close.`,
      ...(p.paper_only ? [`Paper only: it never trades real money. Not backtested yet; this is a forward test.`] : [])],
  };
  const targetText = (p) => (p.target ? `the zone's ${String(p.target).toUpperCase() === "T1" ? "1st" : "2nd"} liquidity level (${String(p.target).toUpperCase()}), or ${p.target_r || 2}× the risk if it has none`
    : `${p.target_r || "?"}× the risk (${p.target_r || "?"}R)`);
  function renderStrategies() {
    const bt = S.backtest || {}, o = bt.overall || {};
    const job = (S.jobs || {}).backtest, running = job && ["queued", "running"].includes(job.state);
    const btn = $("#btn-backtest");
    btn.disabled = !!running; btn.textContent = running ? "Re-testing…" : "Re-test strategies now";
    $("#bt-meta").textContent = bt.from ? `Last re-test: ${bt.from} to ${bt.to}, ${bt.sessions} sessions, ${bt.stocks} stocks, Time Machine ${bt.time_machine ? "on" : "off"}, run ${bt.run_at}.` : "Not re-tested yet.";
    $("#strategy-cards").innerHTML = (S.strategies || []).map((s) => {
      const b = s.backtest || {}, p = s.params || {};
      const prob = /probation/.test(s.why || ""), trial = /paper trial/.test(s.why || "");
      const [cls, label] = !s.enabled ? ["", "Switched off"] : !s.trading ? ["bad", "Not trading today"] : trial ? ["warn", "Trading (paper trial)"] : prob ? ["warn", "Trading (probation)"] : ["good", "Trading today"];
      const how = (HOW[s.name] || (() => []))(p).concat([`Stop-loss at ${stopText(p.stop)}; target ${targetText(p)}.`]);
      return `<div class="card strategy-card"><div class="card-head"><h2>${esc(s.label)}</h2><span class="chip ${cls}">${label}</span></div>
        <ul class="how">${how.map((h) => `<li>${esc(h)}</li>`).join("")}</ul>
        <p class="small ink2"><b>Why:</b> ${esc(s.why || "–")}</p>
        <dl class="kv">
          <dt>Re-test trades</dt><dd>${b.trades ?? "–"}</dd>
          <dt>Winners</dt><dd>${b.win_rate != null ? (100 * b.win_rate).toFixed(0) + "%" : "–"}</dd>
          <dt>Average result after costs</dt><dd class="${pnlClass(b.avg_net_R)}">${b.avg_net_R != null ? (b.avg_net_R >= 0 ? "+" : "") + b.avg_net_R.toFixed(2) + "R" : "–"}</dd>
          <dt>Net profit / loss</dt><dd class="${pnlClass(b.net_pnl)}">${b.net_pnl != null ? signed(b.net_pnl) : "–"}</dd>
          <dt>Worst drop</dt><dd>${b.max_drawdown_inr != null ? inr(b.max_drawdown_inr) : "–"}</dd>
        </dl></div>`;
    }).join("") || `<div class="card"><p class="empty">No strategies configured.</p></div>`;
    const skipped = Object.entries(bt.skipped || {}).map(([k, v]) => `<span class="chip">${esc(k)} ${v}</span>`).join(" ");
    $("#bt-overall").innerHTML = `<dl class="kv">
      <dt>Trades</dt><dd>${o.trades ?? 0}</dd><dt>Winners</dt><dd>${o.win_rate != null ? (100 * o.win_rate).toFixed(0) + "%" : "–"}</dd>
      <dt>Net profit / loss</dt><dd class="${pnlClass(o.net_pnl)}">${signed(o.net_pnl)}</dd><dt>Charges paid</dt><dd>${inr(o.charges)}</dd>
      <dt>Average per trade</dt><dd>${signed(o.expectancy_inr)}</dd><dt>Worst drop</dt><dd>${inr(o.max_drawdown_inr)}</dd>
      <dt>Best / worst day</dt><dd>${signed(o.best_day)} / ${signed(o.worst_day)}</dd></dl>
      <p class="note">Signals not taken: ${skipped || "–"}. About 40 sessions is a small sample; past results do not predict future ones.</p>`;
    lineChart($("#equity"), (o.equity_curve || []).map((r) => ({ x: r.date, y: r.equity, pnl: r.pnl })), (p) => `${p.x}<br>account ${inr(p.y)}<br>day ${signed(p.pnl)}`);
  }

  // ------------------------------------------------------------------ research
  const SETUP_PLAIN = {
    "OR5 stop-order breakout": "Opening-range breakout: a buy (or sell) stop order at the high (or low) of the first 5-minute candle, in the direction that candle closed",
    "Volatility breakout (open +/- 0.5 prev range)": "Volatility breakout: a buy stop at today's open + half of yesterday's range, a sell stop at the open − half",
    "Gap 0.5-2.0% follow": "Gap follow: trade in the direction of a 0.5–2% opening gap",
    "OR15 stop-order breakout": "15-minute opening-range breakout: a buy (or sell) stop order at the high (or low) of 09:15–09:30, live from 09:30",
    "OR30 stop-order breakout": "30-minute opening-range breakout: a buy (or sell) stop order at the high (or low) of 09:15–09:45, live from 09:45",
  };
  const FILTER_PLAIN = (f) => String(f || "").split("+").map((x) => ({
    "top10 rvol": "only the 10 stocks with the most 09:20–09:30 volume vs normal (known at 09:30)",
    "top10 rvol5": "only the 10 stocks with the most 09:15–09:20 volume vs normal ('stocks in play', known at 09:20)",
    nifty: "only in the NIFTY's direction", trend: "only with the stock's trend", rvol: "only on high opening volume",
    "top20 ATR%": "only the 20 most volatile stocks",
  }[x.trim()] || x.trim())).join(", ");
  function exitPlain(b) {
    const p = [];
    const st = String(b.exit_stop || "");
    p.push(st.startsWith("atr") ? `stop-loss ${st.slice(3)} × daily ATR from the entry` : st.startsWith("pct") ? `stop-loss ${st.slice(3)}% away` : st === "struct" ? "stop-loss beyond the setup's own level" : st);
    if (b.exit_target) p.push(`target ${b.exit_target}R`);
    if (b.exit_breakeven) p.push("stop to break-even at +1R");
    if (b.exit_trail) p.push(`trailing stop ${b.exit_trail} × ATR behind the best price`);
    if (b.exit_time_stop) p.push(`out after ${5 * b.exit_time_stop} minutes unless +1R reached`);
    p.push(`otherwise out at ${b.exit_time || "15:15"}`);
    return p.join("; ");
  }
  function loadResearch() {
    return once("research", async () => {
      try { research = await api("/api/research"); } catch (e) { research = { error: e.message }; }
      researchAt = Date.now();
      if (page === "research") renderResearch();
    });
  }
  function renderResearch() {
    if (!research || Date.now() - researchAt > 300000) { loadResearch(); if (!research) return; }
    if (renderResearch.at === researchAt && !renderResearch.force) return;      // static until reloaded
    renderResearch.at = researchAt; renderResearch.force = false;
    if (research.error) { $("#research-body").innerHTML = `<div class="card"><p class="empty">Research unavailable: ${esc(research.error)}</p></div>`; return; }
    const unis = research.universes || [];
    if (!unis.length) { $("#research-body").innerHTML = `<div class="card"><p class="empty">No research results yet. Run <code>./research.sh equity3</code>.</p></div>`; return; }
    const u = unis.find((x) => x.key === rUni) || unis[0];
    const w = u.board[0] || {};
    const tr = w.exact?.train?.per_month, te = w.exact?.test?.per_month;
    const yes = tr >= 20000 && te >= 20000;
    const stress = w.stress || [];
    const st = (k) => stress.find((x) => x.label.toLowerCase().startsWith(k));
    const s5 = st("5 bps"), s8 = st("8 bps"), swc = st("worst");
    const toggle = unis.length > 1 ? `<div class="filter-chips">${unis.map((x) => `<button data-runi="${x.key}" aria-pressed="${x.key === u.key}">${esc(x.label)}</button>`).join("")}</div>` : "";
    const m = w.months_20k || [0, 0, 0, 0], lm = w.months_loss || [0, 0];
    $("#research-body").innerHTML = (research.notice ? `<div class="banner bad" style="margin:0">${esc(research.notice)}</div>` : "") + `
      <div class="card"><div class="card-head"><h2>Can the bot make ₹20,000 a month on ₹1 lakh?</h2><span class="grow"></span>${toggle}</div>
        <div class="answer"><div class="big ${yes ? "yes" : "no"}">${yes ? "Yes*" : "No"}</div>
          <div><p class="lead">${yes ? "On 5-minute bars, yes." : "Not in the backtest."} The best strategy chosen on <b>2021–24 data only</b> (${esc(u.label)}, ${u.stocks || "?"} stocks) made
            <b class="${pnlClass(tr)}">${inr(tr)}</b> a month in 2021–24 and <b class="${pnlClass(te)}">${inr(te)}</b> a month in 2025–26, a period it never saw while being chosen.</p>
          <p><b>Strategy:</b> ${esc(SETUP_PLAIN[w.setup] || w.setup)}; ${esc(FILTER_PLAIN(w.filter))}; ${esc(exitPlain(w))}.</p>
          <p class="small ink2">Worst drop: ${inr(w.exact?.train?.maxdd)} (2021–24) and ${inr(w.exact?.test?.maxdd)} (2025–26) · months of ₹20k or more: ${m[0]} of ${m[1]} and ${m[2]} of ${m[3]} · losing months: ${lm[0]} and ${lm[1]}.</p>
          ${yes ? `<p class="small ink2">* Only with tight execution. ${s5 ? `With 5 bps slippage it made ${inr(s5.train.per_month)} / ${inr(s5.test.per_month)} a month; with 8 bps ${inr(s8.train.per_month)} / ${inr(s8.test.per_month)}; counting every stop touch inside the entry candle as a loss: ${inr(swc.train.per_month)} / ${inr(swc.test.per_month)}.` : ""}</p>` : ""}
          </div></div></div>
      <div class="card flush"><div class="card-head"><h2>The 15 best strategies (picked on 2021–24), full ₹1 lakh account</h2><a href="${esc(u.report)}" target="_blank" rel="noopener">Full report →</a></div>
        <div class="table-wrap"><table><thead><tr><th>#</th><th>Strategy</th><th class="r">2021–24 a month</th><th class="r">2025–26 a month</th><th class="r">Worst drop</th><th class="r">Months ≥ ₹20k</th><th class="r">Trades</th></tr></thead><tbody>
        ${u.board.map((b, i) => `<tr><td>${i + 1}</td><td>${esc(b.setup)}<span class="name" style="max-width:520px;white-space:normal">${esc(FILTER_PLAIN(b.filter))} · ${esc(exitPlain(b))}</span></td>
          <td class="r ${pnlClass(b.exact.train.per_month)}">${inr(b.exact.train.per_month)}</td><td class="r ${pnlClass(b.exact.test.per_month)}"><b>${inr(b.exact.test.per_month)}</b></td>
          <td class="r">${inr(Math.max(b.exact.train.maxdd, b.exact.test.maxdd))}</td><td class="r">${b.months_20k[0]}/${b.months_20k[1]} · ${b.months_20k[2]}/${b.months_20k[3]}</td><td class="r">${NF0.format(b.trades)}</td></tr>`).join("")}
        </tbody></table></div><p class="note" style="padding:0 16px 12px">${esc(research.rules)}</p></div>
      ${u.board.some((b) => b.stress) ? `<div class="card flush"><div class="card-head"><h2>The top 3 under harsher assumptions (2021–24 / 2025–26 a month)</h2></div><div class="table-wrap"><table><thead><tr><th>Strategy</th>${(u.board.find((b) => b.stress).stress).map((x) => `<th class="r">${esc(x.label.replace(/^Base: /, "").replace(/, bar-path rule in the entry bar/, "").replace("Worst case: any touch of the stop in the entry bar is a loss", "worst case in entry candle"))}</th>`).join("")}</tr></thead><tbody>
        ${u.board.filter((b) => b.stress).map((b) => `<tr><td>${esc(b.setup)}<span class="name">${esc(b.filter)}</span></td>${b.stress.map((x) => `<td class="r"><span class="${pnlClass(x.train.per_month)}">${inr(x.train.per_month)}</span> / <b class="${pnlClass(x.test.per_month)}">${inr(x.test.per_month)}</b></td>`).join("")}</tr>`).join("")}
        </tbody></table></div></div>` : ""}
      <div class="grid-2">
        <div class="card"><h3>Everything tested in this round (${esc(u.label)})</h3><div class="table-wrap"><table><thead><tr><th>Family</th><th class="r">Variants</th><th class="r">Profitable 2021–24</th><th class="r">Profitable 2025–26</th><th class="r">≥ ₹20k both</th></tr></thead><tbody>
          ${(u.families || []).map((f) => `<tr><td>${esc({ intraday: "Intraday (MIS)", swing: "Swing (delivery)", pairs: "Sector pairs", etf: "Index ETFs" }[f.family] || f.family)}</td><td class="r">${NF0.format(f.variants)}</td><td class="r">${NF0.format(f.positive_train)}</td><td class="r">${NF0.format(f.positive_test)}</td><td class="r">${f.both_20k}</td></tr>`).join("")}
          </tbody></table></div><p class="note">Family counts use a simpler per-trade model; the leaderboard above uses the full account simulation.</p></div>
        <div class="card"><h3>Read this before trusting any number</h3><ul class="how">
          <li>A backtest is the past. Markets change; the edge in these strategies has been shrinking since 2023.</li>
          <li>The winners depend on getting filled close to the trigger price. Real slippage on busy midcaps can be 5–8 bps or more.</li>
          <li>On 24 Oct 2026 a lookahead flaw was found and fixed: a volume filter known only at 09:30 had been used for entries from 09:20. Every number here is the corrected one.</li>
          <li>The stock lists are today's NIFTY members, which slightly flatters the past.</li>
          <li>The strategies here are research results. The bot's live strategies are on the <a href="#strategies">Strategies</a> page.</li>
          <li>This is research, not investment advice.</li></ul></div></div>
      <div class="card"><div class="card-head"><h2>All research reports</h2></div><ul class="feed">${(research.reports || []).map((r) =>
        `<li><div class="row1"><a href="${esc(r.url)}" target="_blank" rel="noopener"><b>${esc(r.title)}</b></a></div><div class="why">${esc(r.desc)}</div></li>`).join("")}</ul></div>`;
  }

  // ------------------------------------------------------------------ market intel
  async function renderIntel(entering) {
    $$("#intel-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.itab === itab)));
    ["news", "results", "deals"].forEach((t) => $("#intel-" + t).classList.toggle("hidden", t !== itab));
    if (itab === "news") {
      const rows = S.news || [];
      $("#news-list").innerHTML = rows.length ? rows.map((n) => `<li><div class="row1"><span class="num muted">${fmtIST(n.ts, true)}</span>
        <span class="sym" data-sym="${esc(n.symbol)}" tabindex="0">${esc(n.symbol)}</span>
        ${(n.tags || []).map((t) => `<span class="chip ${t === "legal_risk" ? "bad" : ""}">${t === "legal_risk" ? "⚠ legal risk" : esc(t.replace(/_/g, " "))}</span>`).join(" ")}</div>
        <div><a href="${esc(n.link)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a> <span class="muted small">${esc(n.source)}</span></div></li>`).join("")
        : `<li class="empty">No headlines yet for the watchlist (Google News, refreshed every 20 minutes during the session).</li>`;
    } else if (itab === "results") {
      if (entering || !calRows || Date.now() - calAt > 120000) loadCalendar();
      if (resultsSym && !resultsDoc && !busy.has("results")) lookupResults(resultsSym);
    } else if (itab === "deals") {
      if (entering || !dealRows || Date.now() - dealsAt > 120000) loadDeals();
    }
  }
  function loadCalendar() { return once("calendar", loadCalendarNow); }
  async function loadCalendarNow() {
    calAt = Date.now();
    const scope = $("#cal-scope").value;
    try {
      const d = await api(`/api/intel/calendar?scope=${encodeURIComponent(scope)}&days=21`);
      calRows = d.rows;
      $("#cal-table").innerHTML = d.error ? `<p class="empty">${esc(d.error)}</p>` : calRows.length ? `<table><thead><tr><th>Date</th><th>Company</th><th>Board meeting for</th><th></th></tr></thead><tbody>${calRows.map((r) =>
        `<tr class="${r.watch ? "selected" : ""}"><td class="num">${esc(fmtDay(r.date))}</td><td><b>${esc(r.symbol)}</b><span class="name">${esc(r.company)}</span></td>
         <td class="small" title="${esc(r.note)}">${esc(r.purpose)}${r.watch ? ' <span class="chip blue">watchlist</span>' : ""}</td><td><button class="small" data-results="${esc(r.symbol)}">Past results</button></td></tr>`).join("")}</tbody></table>`
        : `<p class="empty">No results announcements in the next 3 weeks for these stocks.</p>`;
    } catch (e) { $("#cal-table").innerHTML = `<p class="empty">${esc(e.message)}</p>`; }
  }
  $("#cal-scope").addEventListener("change", loadCalendar);
  async function lookupResults(sym) {
    sym = String(sym || "").toUpperCase().trim();
    if (!sym) return;
    resultsSym = sym; store.set("mv2.rsym", sym);
    $("#results-form").symbol.value = sym;
    if (page === "intel") renderIntel(false);
    $("#results-view").innerHTML = `<p class="empty">Loading ${esc(sym)}…</p>`;
    await once("results", async () => {
      try { resultsDoc = await api(`/api/intel/results/${encodeURIComponent(sym)}`); }
      catch (e) { resultsDoc = { error: e.message, symbol: sym, quarters: [] }; }
    });
    renderResults();
  }
  function renderResults() {
    const d = resultsDoc;
    if (!d) return;
    if (d.error && !(d.quarters || []).length) { $("#results-view").innerHTML = `<p class="empty">${esc(d.symbol)}: ${esc(d.error)}</p>`; return; }
    const q = (d.quarters || []).slice().reverse();
    const qd = (iso) => new Date(iso + "T00:00:00").toLocaleDateString("en-IN", { month: "short", year: "numeric" });
    const ch = (v) => v == null ? "" : `<span class="small ${pnlClass(v)}">${pct(v)}</span>`;
    $("#results-view").innerHTML = `<h3 style="margin-top:14px">${esc(d.symbol)} · ${esc(d.name || "")}</h3>
      <ul class="how">${(d.summary || []).map((s) => `<li>${esc(s)}</li>`).join("")}</ul>
      <div class="table-wrap"><table><thead><tr><th>Quarter to</th><th class="r">Revenue</th><th class="r">vs year ago</th><th class="r">Net profit</th><th class="r">vs year ago</th><th class="r">EPS</th><th class="r">Operating margin</th><th class="r">Net margin</th></tr></thead><tbody>
      ${q.map((x) => `<tr><td>${esc(qd(x.end))}</td><td class="r">${cr(x.revenue_cr)}</td><td class="r">${ch(x.revenue_cr_yoy)}</td><td class="r ${pnlClass(x.net_income_cr)}">${cr(x.net_income_cr)}</td><td class="r">${ch(x.net_income_cr_yoy)}</td>
        <td class="r">${x.eps != null ? "₹" + NF2.format(x.eps) : "–"}</td><td class="r">${x.operating_margin != null ? x.operating_margin.toFixed(1) + "%" : "–"}</td><td class="r">${x.net_margin != null ? x.net_margin.toFixed(1) + "%" : "–"}</td></tr>`).join("")}
      </tbody></table></div>${d.ttm ? `<p class="note">Last 12 months: revenue ${cr(d.ttm.revenue_cr)}, net profit ${cr(d.ttm.net_income_cr)}, EPS ₹${NF2.format(d.ttm.eps)}.</p>` : ""}
      <p class="note">Source: ${esc(d.source)}. Yahoo keeps about five quarters and sometimes skips one; check the company's filing on NSE before relying on a number.</p>`;
  }
  $("#results-form").addEventListener("submit", (e) => { e.preventDefault(); lookupResults(e.target.symbol.value); });
  function loadDeals() { return once("deals", loadDealsNow); }
  async function loadDealsNow() {
    dealsAt = Date.now();
    const scope = $("#deals-scope").value;
    try {
      const d = await api(`/api/intel/deals?scope=${encodeURIComponent(scope)}&limit=400`);
      dealRows = d.rows;
      $("#deals-asof").textContent = d.as_of ? `latest NSE report: ${fmtDay(d.as_of)}` : "";
      $("#deals-table").innerHTML = d.error && !dealRows.length ? `<p class="empty">${esc(d.error)}</p>` : dealRows.length ? `<table><thead><tr><th>Date</th><th>Stock</th><th>Type</th><th>Who</th><th>Side</th><th class="r">Shares</th><th class="r">Price</th><th class="r">Value</th></tr></thead><tbody>${dealRows.map((r) =>
        `<tr class="${r.watch ? "selected" : ""}"><td class="num">${esc(fmtDay(r.date))}</td><td><span class="sym" data-sym="${esc(r.symbol)}" tabindex="0">${esc(r.symbol)}</span><span class="name">${esc(r.name)}</span></td>
         <td>${r.kind === "block" ? "Block" : "Bulk"}</td><td class="small">${esc(r.client)}</td><td><span class="chip ${r.side === "BUY" ? "good" : r.side === "SELL" ? "bad" : ""}">${esc(r.side || "–")}</span></td>
         <td class="r">${NF0.format(r.qty)}</td><td class="r">${px(r.price)}</td><td class="r">${cr(r.value_cr)}</td></tr>`).join("")}</tbody></table>`
        : `<p class="empty">No deals for these stocks in the last 60 days.</p>`;
    } catch (e) { $("#deals-table").innerHTML = `<p class="empty">${esc(e.message)}</p>`; }
  }
  $("#deals-scope").addEventListener("change", loadDeals);

  // ------------------------------------------------------------------ skills
  async function renderSkills() {
    if (!skills || Date.now() - skillsAt > 10000) {
      if (busy.has("skills")) return;
      let err = null;
      await once("skills", async () => {
        skillsAt = Date.now();
        try { skills = await api("/api/skills"); } catch (e) { err = e; }
      });
      if (err) { $("#skill-cards").innerHTML = `<div class="card"><p class="empty">${esc(err.message)}</p></div>`; return; }
      if (!skills || page !== "skills") return;
    } else return;                                    // nothing new since the last draw
    $("#skill-pipeline").innerHTML = skills.map((s, i) => `${i ? '<span class="arrow">→</span>' : ""}<span class="stage">${esc(s.name)}</span>`).join("");
    $("#skill-cards").innerHTML = skills.map((s) => `<div class="card skill-card">
      <div class="top"><h3 style="margin:0">${esc(s.name)}</h3><span class="grow"></span><span class="chip ${s.active ? "good" : "warn"}">${s.active ? "Running" : "Check"}</span></div>
      <div class="id">${esc(s.id)}</div>
      <p style="margin:0">${esc(s.what)}</p>
      <p class="small ink2" style="margin:0"><b>NSE version:</b> ${esc(s.india)}</p>
      <div class="status-line">${esc(s.status)}</div>
      <div class="foot"><span class="code">${s.code.map(esc).join(" · ")}</span><a href="#${esc(s.page)}">Open →</a></div></div>`).join("");
  }

  // ------------------------------------------------------------------ settings
  function renderSettings() {
    const r = S.rules || {}, f = S.feed || {}, k = S.kill || {}, scan = S.scan || {}, a = S.account || {};
    const live = S.mode === "LIVE";
    $("#set-mode").innerHTML = `<div class="card-head"><h2>Mode</h2><span class="chip ${live ? "bad" : "good"}">${live ? "LIVE · real money" : "Paper · practice money"}</span></div>
      ${live ? `<p>Orders go to your Angel One account. <b>Every new trade waits for your Approve</b> and expires after ${r.approval_timeout_seconds}s; exits (stop-loss, target, flat-by) happen automatically.</p>`
      : `<p>The bot trades by itself with simulated fills at real prices. Nothing reaches your Angel One account.</p>
         <p class="small ink2"><b>To go live</b> (only after weeks of paper results you trust): set <code>TRADING_MODE=LIVE</code> in <code>.env</code> and restart <code>./run.sh</code>. Every entry will then wait for your Approve.</p>`}`;
    const killed = [...Object.entries(k.strategies || {}).map(([n, w]) => `strategy ${n}: ${w}`), ...Object.entries(k.symbols || {}).map(([n, w]) => `${n}: ${w}`)];
    $("#set-safety").innerHTML = `<div class="card-head"><h2>Safety switch</h2><span class="chip ${k.global ? "bad" : "good"}">${k.global ? "New trades paused" : "New trades allowed"}</span></div>
      ${k.global ? `<p>Paused: ${esc(k.global.reason)} <span class="muted small">(${esc(k.global.by)}, ${esc(k.global.at)})</span></p><button class="primary" onclick="document.getElementById('btn-resume').click()">Resume trading</button>`
      : `<p>The bot may open new trades within your rules. Pausing stops new entries only; open positions keep their stop-loss and target.</p><button onclick="document.getElementById('btn-halt').click()">Pause new trades</button>`}
      <p class="small ink2" style="margin-top:10px">Loss limit used today: ${inr(a.risk_used)} of ${inr(a.daily_limit)}. ${killed.length ? "Blocked: " + killed.map(esc).join("; ") : ""}</p>`;
    const rules = [
      ["Capital", `${inr(r.capital_inr)}: the money trades are sized from.`],
      ["Most you lose on one trade", `${inr(r.max_loss_per_trade_inr)}: each trade is sized so hitting its stop-loss loses at most this, before charges.`],
      ["Daily loss limit", `${inr(r.max_daily_loss_inr)}: after losing this much in a day, no new trades until tomorrow.`],
      ["Trades per day", `At most ${r.max_trades_per_day} new trades a day.`],
      ["Open at once", `At most ${r.max_open_positions} positions at the same time.`],
      ["Reward vs risk", `A trade must aim to make at least ${r.min_reward_risk}× what it risks.`],
      ["Entry window", `New trades only between ${esc(r.no_entries_before)} and ${esc(r.no_entries_after)} (India time).`],
      ["Flat by", `Everything still open is closed at ${esc(r.flat_by)}.`],
      ["Position size", `At most ${r.max_position_pct}% of capital in one stock and ${inr(r.max_order_inr)} an order; ${r.cash_buffer_pct}% of cash is kept aside.`],
      ["Leverage", `${r.intraday_leverage}× ${r.intraday_leverage > 1 ? "(borrowed intraday margin)" : "(no borrowed money)"}.`],
      ["Short selling", r.allow_short ? "Allowed (intraday only)." : "Off: the bot only buys."],
      ["Approval timeout", `${r.approval_timeout_seconds}s: an unanswered LIVE trade expires.`],
    ];
    $("#set-rules").innerHTML = rules.map(([k2, v]) => `<li><span class="k">${esc(k2)}</span><span class="v">${v}</span></li>`).join("");
    $("#set-data").innerHTML = `<div class="card-head"><h2>Data &amp; broker</h2></div><dl class="kv">
      <dt>Prices from</dt><dd>${esc(f.mode === "angel" ? "Angel One (real time)" : "Yahoo Finance (delayed)")}</dd><dt>Feed note</dt><dd>${esc(f.note || "–")}</dd>
      <dt>Stocks subscribed</dt><dd>${f.symbols ?? "–"}</dd><dt>Last price update</dt><dd>${f.last_quotes_age_s != null ? Math.round(f.last_quotes_age_s) + "s ago" : "–"}</dd>
      <dt>Broker</dt><dd>${esc(S.broker)}</dd><dt>Morning scan</dt><dd>${scan.scanned ? `${scan.scanned} charts, ${scan.liquid} liquid, for ${esc(scan.trade_date || "–")}` : "pending"}</dd></dl>`;
    const JOBS = { startup: "Start-up", prepare_day: "Morning scan", backtest: "Strategy re-test", news: "News", intel: "Results & deals" };
    const jobs = Object.entries(S.jobs || {}).map(([n, j]) => `<tr><td>${esc(JOBS[n] || n)}</td><td><span class="chip ${j.state === "done" ? "good" : j.state === "failed" ? "bad" : "warn"}">${esc(j.state)}</span></td><td class="small">${esc(j.message || "")}</td></tr>`).join("");
    $("#set-jobs").innerHTML = `<div class="card-head"><h2>Background jobs</h2></div><div class="table-wrap"><table><tbody>${jobs || `<tr><td class="muted">none yet</td></tr>`}</tbody></table></div>
      <div class="form-row" style="margin-top:12px"><button data-job="scan">Run morning scan</button><button data-job="news">Refresh news</button><button data-job="intel/refresh">Refresh results &amp; deals</button><button data-job="backtest">Re-test strategies</button></div>`;
  }

  // ------------------------------------------------------------------ help
  const GLOSSARY = [
    ["Paper mode", "Practice trading: real prices, simulated orders, no real money."],
    ["LIVE mode", "Real orders through Angel One. Every entry waits for your Approve."],
    ["Signal", "The moment a strategy's rule matched. Not every signal becomes a trade."],
    ["Opening range", "The high and low of the first minutes after 09:15. A break of it often starts the day's move."],
    ["VWAP", "Volume-weighted average price: the average price paid today. Above it buyers are in control."],
    ["Stop-loss", "The price at which a losing trade is closed automatically, set before the entry."],
    ["Target", "The price at which a winning trade is closed."],
    ["R", "The amount a trade risks. +2R = made twice the risk; −1R = lost the planned amount."],
    ["ATR", "Average true range: how much a stock usually moves in a day. Used to size stops."],
    ["MIS / intraday", "A position that must be closed the same day. Angel One closes it for you near 15:20 otherwise."],
    ["Flat-by time", "The time the bot closes everything (15:10), before the broker's own square-off."],
    ["Slippage", "The gap between the price you wanted and the price you got. 1 bps = 0.01%."],
    ["Watchlist", "The ~50 stocks the morning scan picked for today."],
    ["Time Machine", "Replays a setup thousands of times on the stock's recent bars to estimate the odds."],
    ["Backtest", "Running a strategy on past data to see how it would have done. The past does not repeat exactly."],
    ["Bulk / block deal", "A big named trade NSE publishes: over 0.5% of a company's shares in a day, or ₹10 crore+ in one go."],
    ["EPS", "Earnings per share: the quarter's profit divided by the number of shares."],
    ["QoQ / YoY", "Quarter-on-quarter (vs the previous quarter) and year-on-year (vs the same quarter a year earlier)."],
  ];
  function renderHelp() {
    const g = $("#glossary");
    if (!g.dataset.done) { g.dataset.done = "1"; g.innerHTML = GLOSSARY.map(([t, d]) => `<div><dt>${esc(t)}</dt><dd>${esc(d)}</dd></div>`).join(""); }
  }

  // ------------------------------------------------------------------ live page: watchlist + chart info
  function nearest(w) {
    let best = null;
    for (const [name, st] of Object.entries(w.strategies || {})) {
      if (st.fired) return { name, fired: true };
      if (st.state === "waiting" && st.distance_pct != null && (!best || Math.abs(st.distance_pct) < Math.abs(best.d))) best = { name, d: st.distance_pct, level: st.level };
    }
    return best;
  }
  const SNAME = { orb: "ORB", level_breakout: "20D high", vwap_reclaim: "VWAP", order_block: "OB", order_block_first: "OB first" };
  function renderLive() {
    const rows = S.watchlist || [];
    const q = ($("#watch-filter").value || "").trim().toUpperCase();
    const soon = new Set(((S.intel || {}).results_soon || []).map((r) => r.symbol));
    const shown = rows.filter((w) => !q || w.symbol.includes(q) || (w.setups || []).some((s) => s.toUpperCase().includes(q)));
    $("#watch-count").textContent = rows.length ? `${rows.length} stocks${S.trade_date ? " · " + fmtDay(S.trade_date) : ""}` : "";
    $("#watch-list").innerHTML = shown.length ? shown.map((w) => {
      const n = nearest(w);
      const trig = n ? (n.fired ? `<span class="chip good">${SNAME[n.name] || n.name} fired</span>` : `<span>${SNAME[n.name] || n.name} trigger ${px(n.level)} (${pct(n.d, 2)})</span>`) : "";
      return `<div class="watch-row ${w.symbol === selected ? "selected" : ""}" data-sym="${esc(w.symbol)}" tabindex="0">
        <div><div class="s">${esc(w.symbol)}</div><div class="n">${esc(w.name)}</div></div>
        <div class="p"><div>${px(w.ltp)}</div><div class="small ${pnlClass(w.change_pct)}">${pct(w.change_pct, 2)}</div></div>
        <div class="meta">${(w.setups || []).slice(0, 3).map((s) => `<span class="chip">${esc(s.replace(/_/g, " "))}</span>`).join("")}${trig}
          ${w.news ? `<span>📰 ${w.news}${(w.news_tags || []).includes("legal_risk") ? " ⚠" : ""}</span>` : ""}${soon.has(w.symbol) ? `<span class="chip blue">results soon</span>` : ""}
          ${w.trade ? `<span class="chip warn">${esc(statusText(w.trade))}</span>` : ""}</div></div>`;
    }).join("") : `<p class="empty">${rows.length ? "No stock matches the filter." : `The watchlist is built by the morning scan (${esc(((S.jobs || {}).prepare_day || {}).state || "starting")}).`}</p>`;
    renderChartInfo();
  }
  $("#watch-filter").addEventListener("input", () => S && renderLive());
  function renderChartInfo() {
    const el = $("#chart-info");
    if (!selected || !S) { el.innerHTML = ""; return; }
    const t = (S.trades || []).find((x) => x.symbol === selected && ACTIVE.includes(x.status)) || (S.trades || []).find((x) => x.symbol === selected);
    const sigs = (S.signals || []).filter((s) => s.symbol === selected).slice(0, 3);
    const w = (S.watchlist || []).find((x) => x.symbol === selected);
    const due = ((S.intel || {}).results_soon || []).find((r) => r.symbol === selected);
    const dg = dealGroups(((S.intel || {}).watch_deals || []).filter((x) => x.symbol === selected))[0];
    const d = chartData && chartData.symbol === selected ? chartData : null;
    el.innerHTML = `
      <div class="box"><h4>Trade</h4>${t ? `${t.side === "buy" ? "Long" : "Short"} ${t.qty} @ ${px(t.entry_price || t.plan.entry)} · stop-loss ${px(t.stop)} · target ${px(t.target)}<br><span class="${pnlClass(t.status === "CLOSED" ? t.net_pnl : t.unrealized)}">${signed(t.status === "CLOSED" ? t.net_pnl : t.unrealized, 2)}</span> · ${esc(statusText(t.status))}` : `<span class="muted">No trade in ${esc(selected)} today.</span>`}</div>
      <div class="box"><h4>Signals today</h4>${sigs.length ? sigs.map((s) => `${esc(s.time)} ${esc(s.label || s.strategy)} ${decisionChip(s.decision)}`).join("<br>") : `<span class="muted">None yet.</span>`}</div>
      <div class="box"><h4>Key levels</h4>${d ? `prev close ${px(d.prev_close)} · opening range ${px(d.or_low)}–${px(d.or_high)} · 20-day high ${px(d.high20)}` : `<span class="muted">–</span>`}</div>
      <div class="box"><h4>News &amp; results</h4>${w && w.news ? `${w.news} headline(s) today. ` : ""}${due ? `<b>Results on ${esc(fmtDay(due.date))}.</b> ` : ""}${dg ? `Big deals on ${esc(fmtDay(dg.date))}: ${esc(dealLine(dg))} ` : ""}<button class="link" data-results="${esc(selected)}">Past results →</button></div>`;
  }

  // ------------------------------------------------------------------ small SVG charts with hover
  function svgFrame(el, h) { const w = Math.max(280, el.clientWidth || 600); return { w, h, pad: { l: 70, r: 12, t: 10, b: 24 } }; }
  function showTip(e, html) {
    const tip = $("#tip");
    tip.innerHTML = html; tip.style.display = "block";
    tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, e.clientX + 12) + "px";
    tip.style.top = (e.clientY + 12) + "px";
  }
  const hideTip = () => ($("#tip").style.display = "none");
  function lineChart(el, pts, fmt) {
    if (!el) return;
    if (pts.length < 2) { el.innerHTML = `<p class="empty">Not enough re-test days to draw.</p>`; return; }
    const { w, h, pad } = svgFrame(el, 220);
    const ys = pts.map((p) => p.y), lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1;
    const X = (i) => pad.l + (i * (w - pad.l - pad.r)) / (pts.length - 1);
    const Y = (v) => pad.t + (h - pad.t - pad.b) * (1 - (v - lo) / span);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
    el.innerHTML = `<svg class="svgchart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Account value over the re-test">
      ${[lo, lo + span / 2, hi].map((t) => `<line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${pad.l - 6}" y="${Y(t) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${inr(t)}</text>`).join("")}
      <text x="${pad.l}" y="${h - 6}" font-size="11" fill="var(--muted)">${esc(pts[0].x)}</text>
      <text x="${w - pad.r}" y="${h - 6}" font-size="11" fill="var(--muted)" text-anchor="end">${esc(pts[pts.length - 1].x)}</text>
      <path d="${d}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>
      <line class="xh" y1="${pad.t}" y2="${h - pad.b}" stroke="var(--axis)" visibility="hidden"/>
      <circle class="xp" r="4" fill="var(--series-1)" stroke="var(--surface)" stroke-width="2" visibility="hidden"/>
      <rect x="${pad.l}" y="0" width="${w - pad.l - pad.r}" height="${h}" fill="transparent" class="hit"/></svg>`;
    const svg = el.querySelector("svg"), hit = svg.querySelector(".hit");
    hit.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * w;
      const i = Math.max(0, Math.min(pts.length - 1, Math.round(((x - pad.l) / (w - pad.l - pad.r)) * (pts.length - 1))));
      const xh = svg.querySelector(".xh"), xp = svg.querySelector(".xp");
      xh.setAttribute("x1", X(i)); xh.setAttribute("x2", X(i)); xh.setAttribute("visibility", "visible");
      xp.setAttribute("cx", X(i)); xp.setAttribute("cy", Y(pts[i].y)); xp.setAttribute("visibility", "visible");
      showTip(e, fmt(pts[i]));
    });
    hit.addEventListener("mouseleave", () => { hideTip(); svg.querySelector(".xh").setAttribute("visibility", "hidden"); svg.querySelector(".xp").setAttribute("visibility", "hidden"); });
  }
  function barChart(el, pts, fmt) {
    if (!el) return;
    if (!pts.length) { el.innerHTML = `<p class="empty">No closed trades yet.</p>`; return; }
    const { w, h, pad } = svgFrame(el, 220);
    const ys = pts.map((p) => p.y), lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), span = hi - lo || 1;
    const Y = (v) => pad.t + (h - pad.t - pad.b) * (1 - (v - lo) / span);
    const slot = (w - pad.l - pad.r) / pts.length, bw = Math.max(2, Math.min(28, slot - 2));
    const bars = pts.map((p, i) => {
      const x = pad.l + i * slot + (slot - bw) / 2, y0 = Y(0), y1 = Y(p.y);
      const top = Math.min(y0, y1), hgt = Math.max(1, Math.abs(y1 - y0));
      return `<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${bw.toFixed(1)}" height="${hgt.toFixed(1)}" rx="${Math.min(4, bw / 2)}" fill="${p.y >= 0 ? "var(--up)" : "var(--down)"}" data-i="${i}"/>
        <rect x="${(pad.l + i * slot).toFixed(1)}" y="0" width="${slot.toFixed(1)}" height="${h}" fill="transparent" data-i="${i}"/>`;
    }).join("");
    el.innerHTML = `<svg class="svgchart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Net profit or loss by day">
      <line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--axis)"/>
      ${[lo, hi].map((t) => `<text x="${pad.l - 6}" y="${Y(t) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${signed(t)}</text>`).join("")}
      ${bars}</svg><p class="note">Blue bars: winning days. Red bars: losing days.</p>`;
    el.querySelectorAll("[data-i]").forEach((r) => {
      r.addEventListener("mousemove", (e) => showTip(e, fmt(pts[+r.dataset.i])));
      r.addEventListener("mouseleave", hideTip);
    });
  }

  // ------------------------------------------------------------------ the price chart
  let chart = null, candles, vwapS, volS, lines = [], chartSym = null, chartData = null;
  function chartTheme() {
    return {
      layout: { background: { color: cssVar("--surface") }, textColor: cssVar("--muted"), fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif" },
      grid: { vertLines: { color: cssVar("--grid") }, horzLines: { color: cssVar("--grid") } },
      rightPriceScale: { borderColor: cssVar("--axis") },
      timeScale: { borderColor: cssVar("--axis"), timeVisible: true, secondsVisible: false },
    };
  }
  function ensureChart() {
    if (chart || !window.LightweightCharts) return;
    const el = $("#chart");
    if (!el.clientWidth) return;               // the page is hidden: create it when shown
    chart = LightweightCharts.createChart(el, Object.assign({ width: el.clientWidth, height: el.clientHeight, crosshair: { mode: 0 } }, chartTheme()));
    candles = chart.addCandlestickSeries({});
    volS = chart.addHistogramSeries({ priceScaleId: "vol", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    vwapS = chart.addLineSeries({ lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    applyChartTheme();
    new ResizeObserver(() => { if (el.clientWidth) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }); }).observe(el);
    chart.subscribeCrosshairMove((p) => {
      const d = p && p.seriesData && p.seriesData.get(candles);
      if (!d) { $("#ohlc").textContent = ""; return; }
      const v = p.seriesData.get(vwapS);
      $("#ohlc").textContent = `${fmtIST(p.time - IST_SHIFT)}  Open ${px(d.open)}  High ${px(d.high)}  Low ${px(d.low)}  Close ${px(d.close)}` + (v ? `  VWAP ${px(v.value)}` : "");
    });
  }
  function applyChartTheme() {
    if (!chart) return;
    chart.applyOptions(chartTheme());
    const up = cssVar("--up"), down = cssVar("--down");
    candles.applyOptions({ upColor: "rgba(0,0,0,0)", borderUpColor: up, wickUpColor: up, downColor: down, borderDownColor: down, wickDownColor: down, borderVisible: true });
    vwapS.applyOptions({ color: cssVar("--series-1") });
    if (chartData) drawChart(chartData, false);
  }
  function selectSymbol(sym, go) {
    sym = String(sym || "").toUpperCase().trim();
    if (!sym) return;
    selected = sym; store.set("mv2.symbol", sym);
    $("#chart-symbol").value = sym;
    if (go && page !== "live") { location.hash = "#live"; return; }
    if (page === "live") { ensureChart(); loadChart(sym, true); renderLive(); }
  }
  $("#chart-symbol").addEventListener("keydown", (e) => { if (e.key === "Enter") selectSymbol(e.target.value, true); });
  async function loadChart(sym, fit) {
    ensureChart();
    if (!chart) return;
    try {
      const d = await api("/api/chart/" + encodeURIComponent(sym));
      if (sym !== selected) return;
      chartData = d;
      drawChart(d, fit || chartSym !== sym);
      chartSym = sym;
      renderChartInfo();
    } catch (e) { $("#chart-source").textContent = "chart unavailable: " + e.message; }
  }
  function drawChart(d, fit) {
    const T = (t) => t + IST_SHIFT;
    const up = cssVar("--up"), down = cssVar("--down"), muted = cssVar("--muted"), ink = cssVar("--ink");
    $("#chart-title").textContent = `${d.symbol}${d.name && d.name !== d.symbol ? " · " + d.name : ""}`;
    $("#chart-source").textContent = d.bars.length ? (d.source === "today" ? (d.forming ? "today, last candle still forming" : "today") : d.source) : "no candles yet";
    candles.setData(d.bars.map((b) => ({ time: T(b.t), open: b.o, high: b.h, low: b.l, close: b.c })));
    volS.setData(d.bars.map((b) => ({ time: T(b.t), value: b.v, color: (b.c >= b.o ? up : down) + "55" })));
    vwapS.setData(d.bars.map((b, i) => ({ time: T(b.t), value: d.vwap[i] })));
    lines.forEach((l) => candles.removePriceLine(l));
    lines = [];
    const add = (price, color, style, title, width = 1) => {
      if (price == null) return;
      lines.push(candles.createPriceLine({ price, color, lineWidth: width, lineStyle: style, axisLabelVisible: true, title }));
    };
    add(d.or_high, muted, 2, "OR high");
    add(d.or_low, muted, 2, "OR low");
    add(d.prev_close, muted, 1, "prev close");
    add(d.high20, muted, 1, "20D high");
    const markers = [];
    const barAt = (ts) => { let hit = null; for (const b of d.bars) { if (b.t <= ts) hit = b; else break; } return hit; };
    for (const t of d.trades || []) {
      if (ACTIVE.includes(t.status)) {
        add(t.entry_price || t.plan.entry, ink, 0, `entry ${t.qty}`, 1);
        add(t.stop, cssVar("--critical"), 0, "stop-loss", 2);
        add(t.target, cssVar("--good"), 0, "target", 2);
      }
      const ob = t.opened_at && barAt(t.opened_at);
      if (ob) markers.push({ time: T(ob.t), position: t.side === "buy" ? "belowBar" : "aboveBar", color: ink, shape: t.side === "buy" ? "arrowUp" : "arrowDown", text: `${t.side === "buy" ? "BUY" : "SELL"} ${t.qty}` });
      const cb = t.closed_at && barAt(t.closed_at);
      if (cb) markers.push({ time: T(cb.t), position: t.side === "buy" ? "aboveBar" : "belowBar", color: muted, shape: "circle", text: `exit (${t.exit_reason})` });
    }
    for (const s of d.signals || []) {
      if (s.decision === "TRADED" || s.decision === "AWAITING_APPROVAL") continue;
      const hit = d.bars.find((b) => fmtIST(b.t + 300) === s.time);
      if (hit) markers.push({ time: T(hit.t), position: s.side === "buy" ? "belowBar" : "aboveBar", color: muted, shape: "square", text: s.decision.toLowerCase() });
    }
    markers.sort((a, b) => a.time - b.time);
    candles.setMarkers(markers);
    if (fit) chart.timeScale().fitContent();
  }

  // ------------------------------------------------------------------ order-block board
  let ob = null, obAt = 0, obTab = store.get("mv2.obtab", "board"), obSel = store.get("mv2.obsym", null);
  let obDetail = null, obDetailAt = 0;
  let obChart = null, obCandles = null, obVwap = null, obSeries = [], obLines = [], obChartSym = null, obChartLastT = 0;
  // live: every second, prices, live OI and the price-based confirmations; ranks, zones and alignment move each minute
  let obLive = null;
  const obFl = {}, obPrevPx = {};          // a price cell flashes when the price changes
  const obOpen = () => ((S && S.clock) || {}).state === "OPEN";
  const obStale = () => Date.now() - obAt > (obOpen() ? 15000 : 120000);
  const sideCls = (s) => (s > 0 ? "side-bull" : s < 0 ? "side-bear" : "muted");
  const sideWord = (s) => (s > 0 ? "Bullish" : s < 0 ? "Bearish" : "Flat");
  const ZCHIP = { fresh: ["good", "fresh"], retested: ["warn", "retested"], broken: ["bad", "broken"] };
  // stop-loss = the zone's invalidation; target = T2 (the last liquidity level), else 2R: what the paper trade uses
  const zTarget = (z) => (z.targets.length ? z.targets[z.targets.length - 1] : z.z + 2 * (z.z - z.invalid));
  const zTargets = (z) => (z.targets.length ? z.targets.map((t, i) => `T${i + 1} ${px(t)}`).join(" · ") : `${px(zTarget(z))} (2R: no level ahead)`);
  const zRisk = (z) => `₹${px(Math.abs(z.z - z.invalid))} a share, ${(100 * Math.abs(z.z - z.invalid) / z.z).toFixed(2)}%`;
  const rr = (z) => (!z ? "–" : z.rr != null ? `1:${z.rr.toFixed(1)}` : "1:2.0");
  // Z is given when the break candle closes; the entry signal is the first candle back at Z (entry_at = its start)
  const zGiven = (z) => (z && z.known_at ? fmtIST(z.known_at) : "–");
  const zEntry = (z) => (z && z.entry_at ? fmtIST(z.entry_at) : null);
  const zWhen = (z) => `given ${zGiven(z)} · ${zEntry(z) ? "entry " + zEntry(z) : "no entry yet"}`;
  const PCHIP = { traded: "good", armed: "good", idle: "", off: "", skipped: "warn", blocked: "bad", refused: "warn", benched: "bad", stale: "warn" };
  // two paper strategies trade the board: the retest (ob.paper) and the first stage (ob.paper_first)
  const paperOf = (sym, k = "paper") => (((ob && ob[k]) || {}).rows || {})[sym];
  const paperLine = (sym, k = "paper") => { const x = paperOf(sym, k); return x ? `<span class="chip ${PCHIP[x.state] || ""}">${k === "paper" ? "Paper · retest" : "Paper · first stage"}</span> ${esc(x.text)}` : ""; };
  // targets as given: the levels when the zone was given (they stay); next levels now: ahead of the price (they move)
  const tList = (t) => (t && t.length ? t.map((v, i) => `T${i + 1} ${px(v)}`).join(" · ") : null);
  const zGivenTargets = (z) => tList(z.given_targets) || `${px(z.z + 2 * (z.z - z.invalid))} (2R: no level ahead)`;
  const firstWin = () => { const s = ((S && S.strategies) || []).find((x) => x.name === "order_block_first"); return 300 * ((s && s.params && s.params.window_bars) || 1); };
  // heads-up (the board, every second): a break forming, a zone just given (the first-stage plan), the price near Z
  const HEADS = { break: ["blue", (s) => (s > 0 ? "breaking out" : "breaking down")], given: ["good", () => "zone given · first stage"], near: ["warn", () => "near Z"] };
  const obHeadFl = {}, obHeadKey = {};    // a heads-up chip flashes once when it appears
  const headChip = (r) => { const k = r.heads && HEADS[r.heads.kind]; return k ? `<span class="chip ${k[0]} heads ${obHeadFl[r.symbol] ? "flash-up" : ""}">${esc(k[1](r.side))}</span>` : ""; };
  function headText(r) {
    const h = r.heads, buy = r.side > 0;
    if (!h) return "";
    if (h.kind === "break") return `₹${px(h.price)} is ${buy ? "above" : "below"} the swing ${buy ? "high" : "low"} ${px(h.level)}: if the ${fmtIST(h.closes - 300)} candle closes there, Z ${px(h.z)} · SL ${px(h.stop)}${h.t.length ? " · " + tList(h.t) : ""}`;
    if (h.kind === "given") return `first stage now ₹${px(h.price)} · SL ${px(h.stop)} · target ${px(h.target)} · R:R 1:${h.rr.toFixed(1)} · until ${fmtIST(h.until)}${h.gate ? ` · the risk check would refuse: ${h.gate}` : " · passes the price checks"}`;
    return `₹${px(h.dist)} (${h.dist_atr.toFixed(2)} ATR) from Z ${px(h.z)}, not touched yet`;
  }
  // today's heads-up log (the server's), polled from /api/state on every page; a beep and a notification for new ones
  let obAlerts = [], obAlertSeq = null, obAlertDay = null, obAlertsBusy = false, audio = null;
  const obAlertsOn = () => store.get("mv2.obalerts", "0") === "1";
  async function obAlertsPoll() {
    const a = S && S.ob_alerts;
    if (!a || obAlertsBusy) return;
    const first = obAlertSeq === null || a.seq < obAlertSeq;     // first look, or the engine restarted
    if (!first && a.seq === obAlertSeq) return;
    obAlertsBusy = true;
    try {
      const r = await api("/api/ob/alerts?after=" + (first ? 0 : obAlertSeq));
      obAlerts = first || r.day !== obAlertDay ? r.alerts : obAlerts.concat(r.alerts).slice(-300);
      obAlertSeq = r.seq; obAlertDay = r.day;
      if (!first && r.alerts.length && obAlertsOn()) { beep(); r.alerts.forEach(notifyAlert); }
      if (page === "ob" && obTab === "board" && ob && ob.ready) keepPlace($("#ob-board"), drawOBBoard);
    } catch (e) { /* the next poll tries again */ } finally { obAlertsBusy = false; }
  }
  function beep() {
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      if (audio.state === "suspended") audio.resume();
      const o = audio.createOscillator(), g = audio.createGain(), t = audio.currentTime;
      o.type = "sine"; o.frequency.setValueAtTime(880, t); o.frequency.setValueAtTime(1175, t + 0.12);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.2, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.3);
      o.connect(g); g.connect(audio.destination); o.start(t); o.stop(t + 0.32);
    } catch (e) { /* no sound in this browser */ }
  }
  function notifyAlert(a) {
    try {
      if (!window.Notification || Notification.permission !== "granted") return;
      const n = new Notification(`MONEYV2 · ${a.sym}`, { body: a.text, tag: "mv2-ob-" + a.seq });
      n.onclick = () => { window.focus(); obSel = a.sym; store.set("mv2.obsym", a.sym); location.hash = "#ob"; n.close(); };
    } catch (e) { /* notifications blocked */ }
  }
  async function toggleAlerts() {
    const on = !obAlertsOn();
    store.set("mv2.obalerts", on ? "1" : "0");
    if (on) {
      beep();                                       // this click lets the page play sound
      if (window.Notification && Notification.permission === "default") { try { await Notification.requestPermission(); } catch (e) { /* old browser */ } }
      toast(window.Notification && Notification.permission === "granted" ? "Heads-up alerts on: a beep and a notification for each new one"
        : "Heads-up alerts on: a beep for each new one (this browser blocks notifications for this page: allow them in its site settings)");
    } else toast("Heads-up alerts off");
    if (page === "ob" && ob && ob.ready) keepPlace($("#ob-board"), drawOBBoard);
  }
  // after a reload, sound needs one click or key press on the page before it can play
  const unlockAudio = () => { if (obAlertsOn() && (!audio || audio.state === "suspended")) { try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); audio.resume(); } catch (e) { /* no sound */ } } };
  document.addEventListener("pointerdown", unlockAudio, { passive: true });
  document.addEventListener("keydown", unlockAudio);
  function firstStage(r, z) {
    if (!z) return "–";
    if (r.heads && r.heads.kind === "given") return `<b>now</b>: ${esc(headText(r))}`;
    if (z.entry_at) return `over: the price came back to Z at ${zEntry(z)} (the retest's trade)`;
    const end = z.known_at + firstWin();
    return Date.now() / 1000 > end ? `over: it was ${zGiven(z)}–${fmtIST(end)}, right after the zone was given`
      : `until ${fmtIST(end)}: near the live price · SL ${px(z.invalid)} · aiming for ${zGivenTargets(z)}`;
  }
  function loadOB() {
    return once("ob", async () => {
      try { ob = await api("/api/ob/board"); } catch (e) { ob = { ready: false, error: e.message }; }
      obAt = Date.now();
    });
  }
  function renderOB(entering) {
    $$("#ob-tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.obtab === obTab)));
    ["board", "confirm", "map", "sectors"].forEach((k) => $("#ob-" + k).classList.toggle("hidden", obTab !== k));
    if (entering) drawOB();
    if (entering || obStale()) loadOB().then(() => { if (page === "ob") drawOB(); });
  }
  function obRows() { return ob && ob.ready ? [...(ob.bullish || []), ...(ob.bearish || [])] : []; }
  const obLastT = () => { const b = obDetail && obDetail.bars; return b && b.length ? b[b.length - 1].t : 0; };
  function obMerge() {
    const L = obLive;
    if (!L || !L.ready || !ob || !ob.ready || L.board_at !== ob.updated_at) return false;
    const seen = new Set();
    for (const r of [...(ob.picks || []), ...obRows()]) {
      const x = L.rows[r.symbol];
      if (!x) continue;
      r.live_last = x.last; r.change_pct = x.chg;
      if (x.confirm) r.confirm = x.confirm;
      r.heads = x.heads || null;
      if (seen.has(r.symbol)) continue;
      seen.add(r.symbol);
      const was = obPrevPx[r.symbol];
      obFl[r.symbol] = was == null || x.last === was ? "" : x.last > was ? "flash-up" : "flash-down";
      obPrevPx[r.symbol] = x.last;
      const hk = x.heads ? `${x.heads.kind}:${x.heads.closes || x.heads.formed}` : "";
      obHeadFl[r.symbol] = !!hk && hk !== obHeadKey[r.symbol];
      obHeadKey[r.symbol] = hk;
    }
    for (const p of ob.map || []) if (L.chg[p.symbol] != null) p.change_pct = L.chg[p.symbol];
    if (L.index_chg != null) ob.index_chg = L.index_chg;
    if (L.live_oi) ob.live_oi = Object.assign({}, ob.live_oi, L.live_oi);
    return true;
  }
  function obTick() {
    if (page !== "ob" || document.hidden || !ob || !ob.ready || !ob.live) return;
    const d = obDetail, q = d && !d.error && d.symbol === obSel ? `?sym=${encodeURIComponent(obSel)}&after=${obLastT()}` : "";
    once("obl", async () => {
      try { obLive = await api("/api/ob/live" + q); } catch (e) { return; }
      if (!obLive.ready || page !== "ob") return;
      if (obLive.board_at !== ob.updated_at) {               // a new minute's board: ranks, zones and candles
        await loadOB();
        obMerge();
        drawOB(true);
        loadOBDetail(obSel, true);
        return;
      }
      if (obMerge()) drawOB(true);
    });
  }
  setInterval(obTick, 1000);
  // redraw a container without losing its tables' sideways scroll or the focused row
  function keepPlace(el, draw) {
    const xs = [...el.querySelectorAll(".table-wrap")].map((w) => w.scrollLeft);
    const f = document.activeElement && el.contains(document.activeElement) ? document.activeElement.dataset.ob : null;
    draw();
    el.querySelectorAll(".table-wrap").forEach((w, i) => { if (xs[i]) w.scrollLeft = xs[i]; });
    if (f) { const n = el.querySelector(`[data-ob="${CSS.escape(f)}"]`); if (n) n.focus({ preventScroll: true }); }
  }
  function drawOB(fromTick) {
    const head = $("#ob-head");
    if (!ob) { head.innerHTML = `<div class="card"><p class="empty">Loading the board…</p></div>`; return; }
    if (!ob.ready) { head.innerHTML = `<div class="card"><p class="empty">${esc(ob.error || "The board is being built.")}</p></div>`; $("#ob-board").innerHTML = ""; return; }
    const sess = ob.live ? `<span class="chip good">Live</span> prices every second · ranks and zones ${fmtIST(ob.updated_at)}` : `<span class="chip">Last session</span> ${esc(fmtDay(ob.session))} close`;
    const tot = Math.max(1, ob.up + ob.down);
    head.innerHTML = `<div class="card"><div class="ob-status">
        <div>${sess} · ${ob.stocks} F&amp;O stocks${ob.fno_date ? ` · OI as of ${esc(fmtDay(ob.fno_date))}` : ""}</div>
        <div class="breadth" title="F&O stocks up and down on the day"><span class="side-bull">${ob.up} up</span>
          <div class="bar"><span style="width:${(100 * ob.up / tot).toFixed(1)}%;background:var(--up)"></span><span style="width:${(100 * ob.down / tot).toFixed(1)}%;background:var(--down)"></span></div>
          <span class="side-bear">${ob.down} down</span></div>
        ${ob.leading ? `<div class="small">Leading: <b>${esc(ob.leading.sector)}</b> ${pct(ob.leading.avg_change, 2)} · Lagging: <b>${esc(ob.lagging.sector)}</b> ${pct(ob.lagging.avg_change, 2)}</div>` : ""}
      </div></div>`;
    if (!obSel || !obRows().some((r) => r.symbol === obSel)) {
      const first = (ob.picks || [])[0] || obRows()[0];
      obSel = first ? first.symbol : null;
    }
    const draw = { board: drawOBBoard, confirm: drawOBConfirm, map: drawOBMap, sectors: drawOBSectors }[obTab];
    if (draw) keepPlace($("#ob-" + obTab), draw);
    if (fromTick) { obDetailLive(); for (const k in obFl) obFl[k] = ""; for (const k in obHeadFl) obHeadFl[k] = false; }
    else loadOBDetail(obSel, false);
  }
  const obPx = (r) => `<span class="num ${obFl[r.symbol] || ""}">₹${px(r.live_last != null ? r.live_last : r.last)}</span>`;
  function pickCard(r) {
    const z = r.zone;
    return `<div class="pick ${r.symbol === obSel ? "selected" : ""}" data-ob="${esc(r.symbol)}" tabindex="0">
      <div class="top"><b>${esc(r.symbol)}</b><span class="grow"></span><span class="${pnlClass(r.change_pct)} num">${pct(r.change_pct, 2)}</span></div>
      <div class="small"><span class="${sideCls(r.side)}">${sideWord(r.side)}</span> · ${obPx(r)}</div>
      <div class="muted small">${esc(r.sector)} · #${r.rank} · ${r.aligned}/4 aligned${r.confirm ? ` · ${r.confirm.score}/${r.confirm.of} confirmed` : ""}</div>
      <div class="lv">${z ? `<span>${r.side < 0 ? "Sell zone" : "Buy zone"}</span><span>${px(z.bottom)} – ${px(z.top)} (${esc(z.status)})</span>
        <span>${r.side < 0 ? "Sell at Z" : "Buy at Z"}</span><span><b>${px(z.z)}</b> · given ${zGiven(z)}</span>
        <span>Stop-loss</span><span><b class="neg">${px(z.invalid)}</b></span>
        <span>Target as given</span><span><b class="pos">${zGivenTargets(z)}</b></span>
        <span>Next levels</span><span>${zTargets(z)}</span><span>R:R</span><span>${rr(z)}</span>
        <span>Entry signal</span><span>${zEntry(z) ? `back at Z ${zEntry(z)}` : (ob.live ? "waiting for price to return" : "price never returned")}</span>`
        : `<span>Zone</span><span>no order block yet</span>`}</div>${r.heads ? `<div class="small heads-line">${headChip(r)} ${esc(headText(r))}</div>` : ""}
      ${["paper", "paper_first"].map((k) => (paperOf(r.symbol, k) ? `<div class="small paper-line">${paperLine(r.symbol, k)}</div>` : "")).join("")}</div>`;
  }
  function alignDots(r) {
    const c = r.checks || {};
    const lab = { stock: "in the top 20 on its side", sector: "its sector moves the same way", breadth: "most F&O stocks move the same way", vwap: "on the right side of VWAP" };
    return `<span class="align" title="${Object.keys(lab).map((k) => (c[k] ? "✓ " : "✗ ") + lab[k]).join("\n")}">${Object.keys(lab).map((k) => `<i class="${c[k] ? "on" : ""}"></i>`).join("")}</span>`;
  }
  function obTable(rows, side) {
    if (!rows.length) return `<p class="empty">No ${side > 0 ? "bullish" : "bearish"} stock is moving cleanly ${ob.live ? "right now" : "in this session"}.</p>`;
    const mx = Math.max(...rows.map((r) => Math.abs(r.score || 0)), 0.01);
    return `<div class="table-wrap"><table class="ob-table"><thead><tr><th>#</th><th>Stock</th><th class="r">Change</th><th>Strength</th><th>Entry · stop-loss · target</th><th>Status</th></tr></thead><tbody>${rows.map((r) => {
      const z = r.zone, zc = z ? ZCHIP[z.status] : null;
      const d = r.rank_change ? `<span class="delta ${r.rank_change > 0 ? "up" : "down"}">${r.rank_change > 0 ? "▲" : "▼"}${Math.abs(r.rank_change)}</span>` : "";
      const rule = r.rule === "pick" ? `<span class="chip good">Pick</span>` : r.event ? `<span class="chip warn">Event risk</span>` : `<span class="muted small">${esc((r.rule || "").replace("not picked · ", ""))}</span>`;
      return `<tr class="clickable ${r.symbol === obSel ? "selected" : ""} ${r.new_top10 ? "top10" : ""}" data-ob="${esc(r.symbol)}" tabindex="0">
        <td class="num">${r.rank}${d}</td><td><b>${esc(r.symbol)}</b><span class="name">${esc(r.sector)}${r.oi_buildup ? " · OI: " + esc(r.oi_buildup) : ""}</span></td>
        <td class="r ${pnlClass(r.change_pct)}">${pct(r.change_pct, 2)}<span class="name">${obPx(r)}</span></td>
        <td class="num"><span class="sbar" style="width:${(42 * Math.abs(r.score) / mx).toFixed(0)}px;background:${side > 0 ? "var(--up)" : "var(--down)"}"></span>${(r.score || 0).toFixed(2)}<span class="name">p${r.pctile} · straight ${Math.round(100 * (r.clean || 0))}%</span></td>
        <td>${headChip(r)}${z ? `<span class="chip ${zc[0]}">${zc[1]}</span><span class="name num">Z ${px(z.z)} · ${rr(z)}</span><span class="name num">SL <span class="neg">${px(z.invalid)}</span> · T <span class="pos">${px(zTarget(z))}</span></span><span class="name num">${zWhen(z)}</span>` : `<span class="muted small">${r.event ? "no levels" : "none yet"}</span>`}${r.heads ? `<span class="name" title="${esc(headText(r))}">${esc(headText(r))}</span>` : ""}</td>
        <td>${rule}<span class="name">${alignDots(r)} ${r.aligned}/4</span></td></tr>`;
    }).join("")}</tbody></table></div>`;
  }
  function obAlertsCard() {
    const on = obAlertsOn(), rows = obAlerts.slice(-40).reverse();
    return `<div class="card flush ob-alerts"><div class="card-head"><h2>Heads-up</h2>
        <span class="muted small">a break forming · a zone just given (first stage) · the price near Z · today, newest first</span>
        <span class="grow"></span><button class="small" data-obalerts="1" aria-pressed="${on}">${on ? "Alerts on: beep + notification" : "Alerts off: turn on"}</button></div>
      ${rows.length ? `<div class="table-wrap"><table class="ob-table"><tbody>${rows.map((a) => {
        const k = HEADS[a.kind] || ["", () => a.kind];
        return `<tr class="clickable" data-ob="${esc(a.sym)}" tabindex="0"><td class="num">${fmtIST(a.at)}</td><td><b>${esc(a.sym)}</b></td>
          <td><span class="chip ${k[0]}">${esc(k[1](a.side))}</span></td><td class="small wrap">${esc(a.text)}</td></tr>`;
      }).join("")}</tbody></table></div>`
        : `<p class="empty">${ob.live ? `Nothing yet today. Each heads-up is listed here once${on ? ", with a beep and a notification" : "; turn Alerts on for a beep and a notification"}.` : "Heads-up runs during the session."}</p>`}</div>`;
  }
  function drawOBBoard() {
    const picks = ob.picks || [];
    $("#ob-board").innerHTML = `${obAlertsCard()}${picks.length ? `<div class="ob-picks">${picks.map(pickCard).join("")}</div>` : `<div class="card"><p class="empty">No picks: nothing passed the rules${ob.live ? " yet" : " this session"}.</p></div>`}
      <div class="ob-boards">
        <div class="card flush"><div class="card-head"><h2 class="side-bull">Bullish board</h2><span class="muted small">top 20 by strength · for buyers</span></div>${obTable(ob.bullish || [], 1)}</div>
        <div class="card flush"><div class="card-head"><h2 class="side-bear">Bearish board</h2><span class="muted small">top 20 by strength · for sellers</span></div>${obTable(ob.bearish || [], -1)}</div>
      </div>`;
  }
  const cfMark = (k) => (!k || k.ok == null ? `<span class="cf-mark na">–</span>` : k.ok ? `<span class="cf-mark on">✓</span>` : `<span class="cf-mark off">✗</span>`);
  function cfTable(rows, side) {
    const rules = ob.confirm_rules || [];
    rows = rows.filter((r) => r.confirm).sort((a, b) => b.confirm.score - a.confirm.score || a.rank - b.rank);
    if (!rows.length) return `<p class="empty">No ${side > 0 ? "bullish" : "bearish"} stock is moving cleanly ${ob.live ? "right now" : "in this session"}.</p>`;
    return `<div class="table-wrap"><table class="ob-table cf-table"><thead><tr><th>Stock</th><th class="r">Change</th><th>Confirmed</th>
      ${rules.map((x) => `<th class="cf" title="${esc(x.rule)}">${esc(x.label)}</th>`).join("")}<th>Board</th><th>Zone</th></tr></thead><tbody>${rows.map((r) => {
        const c = r.confirm, z = r.zone, zc = z ? ZCHIP[z.status] : null;
        return `<tr class="clickable ${r.symbol === obSel ? "selected" : ""}" data-ob="${esc(r.symbol)}" tabindex="0">
          <td><b>${esc(r.symbol)}</b><span class="name">#${r.rank} · ${esc(r.sector)}</span></td>
          <td class="r ${pnlClass(r.change_pct)}">${pct(r.change_pct, 2)}<span class="name">${obPx(r)}</span></td>
          <td class="num"><span class="sbar" style="width:${(42 * c.score / Math.max(1, rules.length)).toFixed(0)}px;background:${side > 0 ? "var(--up)" : "var(--down)"}"></span>${c.score}/${c.of}</td>
          ${rules.map((x) => { const k = c.checks[x.key]; return `<td class="cf" title="${esc(x.label + ": " + ((k && k.text) || "no data"))}">${cfMark(k)}<span class="name num">${esc((k && k.v) || "–")}</span></td>`; }).join("")}
          <td>${alignDots(r)} <span class="small">${r.aligned}/4</span>${r.rule === "pick" ? ` <span class="chip good">Pick</span>` : ""}</td>
          <td>${z ? `<span class="chip ${zc[0]}">${zc[1]}</span><span class="name num">Z ${px(z.z)}</span>` : `<span class="muted small">${r.event ? "event risk" : "none yet"}</span>`}</td></tr>`;
      }).join("")}</tbody></table></div>`;
  }
  function drawOBConfirm() {
    const lo = ob.live_oi || {}, lp = ob.live_opt || {};
    const oi = lo.streamed && lo.stream_at ? `Live futures OI: ${lo.stocks} stocks from Angel One · ${lo.streamed} contracts streaming (last change ${fmtIST(lo.stream_at)}), the rest every 3 minutes`
      : lo.at ? `Live futures OI: ${lo.stocks} stocks from Angel One, read ${fmtIST(lo.at)}`
      : `Live futures OI: ${lo.error ? esc(lo.error) : ob.live ? "first reading within a few minutes" : "only during the session"}`;
    const op = lp.at ? `Live options OI: ${lp.stocks} leaders, nearest expiry, read ${fmtIST(lp.at)} (every 3 minutes)`
      : `Live options OI: ${lp.error ? esc(lp.error) : ob.live ? "first reading within a few minutes" : "only during the session"}`;
    $("#ob-confirm").innerHTML = `<div class="card"><div class="ob-status small">
        <div>Futures and options OI: NSE bhavcopy ${ob.fno_date ? esc(fmtDay(ob.fno_date)) : "not available"}</div><div>${oi}</div><div>${op}</div>
        ${ob.index_chg != null ? `<div>NIFTY 50 <span class="${pnlClass(ob.index_chg)}">${pct(ob.index_chg, 2)}</span> from its open</div>` : ""}</div>
        <p class="note" style="margin:0">Each check asks whether that piece of evidence agrees with the stock's side on the board: ✓ agrees, ✗ disagrees, – no data (not counted).
          ${ob.live ? "Prices, OI today, volume, CPR, previous day, trend and NIFTY are re-checked every second; options today every 3 minutes; yesterday's OI once a day. " : ""}Hover a column for its rule, a cell for its numbers. Sorted by confirmations, then board rank. These checks are not part of the picks and are not backtested.</p></div>
      ${obBuildups()}
      <div class="card flush"><div class="card-head"><h2 class="side-bull">Bullish confirmations</h2><span class="muted small">the bullish board's top 20</span></div>${cfTable(ob.bullish || [], 1)}</div>
      <div class="card flush"><div class="card-head"><h2 class="side-bear">Bearish confirmations</h2><span class="muted small">the bearish board's top 20</span></div>${cfTable(ob.bearish || [], -1)}</div>`;
  }
  const qty = (v) => { if (v == null || isNaN(v)) return "–"; const a = Math.abs(v); return (v < 0 ? "−" : "") + (a >= 1e7 ? (a / 1e7).toFixed(1) + " Cr" : a >= 1e5 ? (a / 1e5).toFixed(1) + " L" : a >= 1e3 ? (a / 1e3).toFixed(0) + "k" : a.toFixed(0)); };
  const BUILDS = [["long build-up", "side-bull", "price up, OI up"], ["short covering", "side-bull", "price up, OI down"],
    ["short build-up", "side-bear", "price down, OI up"], ["long unwinding", "side-bear", "price down, OI down"]];
  function obBuildups() {
    const b = ob.oi_lists || {}, L = b.lists || {};
    const head = `<div class="card-head"><h2>OI build-up</h2><span class="muted small">Angel One's lists, near-month stock futures${b.at ? ` · read ${fmtIST(b.at)}` : ""}</span></div>`;
    if (!Object.keys(L).length) return `<div class="card">${head}<p class="empty">${esc(b.error || (ob.live ? "The first lists come within a few minutes." : "Angel One's lists are read during the session."))}</p></div>`;
    const onBoard = new Set(obRows().map((r) => r.symbol));
    return `<div class="card flush">${head}<div class="oi-lists">${BUILDS.map(([k, cls, what]) => `<div><h3><span class="${cls}">${esc(k)}</span> <span class="muted small">${what}</span></h3>
      ${(L[k] || []).length ? `<table class="ob-table"><tbody>${L[k].map((x) => `<tr${onBoard.has(x.symbol) ? ` class="clickable" data-ob="${esc(x.symbol)}" tabindex="0"` : ""}>
        <td><b>${esc(x.symbol)}</b>${onBoard.has(x.symbol) ? `<span class="name">on the board</span>` : ""}</td><td class="r ${pnlClass(x.chg_pct)}">${pct(x.chg_pct, 2)}</td>
        <td class="r num" title="near-month futures open interest (shares) and its change">OI ${qty(x.oi)}<span class="name ${pnlClass(x.oi_chg_pct)}">${x.oi_chg_pct == null ? "–" : pct(x.oi_chg_pct, 1)}</span></td></tr>`).join("")}</tbody></table>`
        : `<p class="empty">none</p>`}</div>`).join("")}</div>${b.error ? `<p class="note" style="padding:0 16px 12px">${esc(b.error)}</p>` : ""}</div>`;
  }
  function drawOBSectors() {
    const rows = ob.sectors || [];
    const mx = Math.max(...rows.map((r) => Math.abs(r.avg_score)), 0.01);
    $("#ob-sectors").innerHTML = `<div class="card flush"><div class="card-head"><h2>Sector flow</h2><span class="muted small">sectors ranked by the average strength of their F&amp;O stocks</span></div>
      <div class="table-wrap"><table><thead><tr><th>Sector</th><th class="r">Stocks</th><th class="r">Up</th><th class="r">Avg change</th><th>Avg strength</th></tr></thead><tbody>${rows.map((r) =>
        `<tr><td><b>${esc(r.sector)}</b>${ob.leading && r.sector === ob.leading.sector ? ' <span class="chip good">leading</span>' : ""}${ob.lagging && r.sector === ob.lagging.sector ? ' <span class="chip bad">lagging</span>' : ""}</td>
        <td class="r">${r.stocks}</td><td class="r">${r.up} of ${r.stocks}</td><td class="r ${pnlClass(r.avg_change)}">${pct(r.avg_change, 2)}</td>
        <td class="num"><span class="sbar" style="width:${(60 * Math.abs(r.avg_score) / mx).toFixed(0)}px;background:${r.avg_score >= 0 ? "var(--up)" : "var(--down)"}"></span>${r.avg_score.toFixed(2)}</td></tr>`).join("")}</tbody></table></div>
      <p class="note" style="padding:0 16px 14px">A 0.5% sector move with 11 of 12 stocks up is a real sector move; with 3 of 12 up, a few stocks are dragging the average.</p></div>`;
  }
  function drawOBMap() {
    const el = $("#ob-map");
    const pts = (ob.map || []).filter((p) => p.change_pct != null);
    if (!pts.length) { el.innerHTML = `<div class="card"><p class="empty">No stocks to plot.</p></div>`; return; }
    const secs = [...new Set(pts.map((p) => p.sector))];
    const avg = (s) => { const q = pts.filter((p) => p.sector === s); return q.reduce((a, p) => a + p.change_pct, 0) / q.length; };
    secs.sort((a, b) => avg(b) - avg(a));
    const sorted = pts.map((p) => Math.abs(p.change_pct)).sort((a, b) => a - b);
    const lim = Math.max(1, Math.ceil(sorted[Math.floor(0.98 * (sorted.length - 1))]));
    const w = Math.max(320, (el.clientWidth || 900) - 34), rowH = 22, pad = { l: 190, r: 70, t: 26, b: 26 };
    const h = pad.t + pad.b + rowH * secs.length;
    const X = (v) => pad.l + ((Math.max(-lim, Math.min(lim, v)) + lim) / (2 * lim)) * (w - pad.l - pad.r);
    const ticks = [-lim, -lim / 2, 0, lim / 2, lim];
    let dots = "";
    secs.forEach((s, i) => {
      const y0 = pad.t + i * rowH + rowH / 2;
      pts.filter((p) => p.sector === s).forEach((p, j) => {
        const jit = ((j * 7) % 9 - 4) * 1.2;
        const fill = p.event ? "var(--warning)" : p.change_pct >= 0 ? "var(--up)" : "var(--down)";
        const r = p.contender ? 5 : 3.5;
        dots += `<circle cx="${X(p.change_pct).toFixed(1)}" cy="${(y0 + jit).toFixed(1)}" r="${r}" fill="${fill}" fill-opacity="${p.contender ? 0.95 : 0.55}"
          stroke="${p.pick ? "var(--ink)" : "var(--surface)"}" stroke-width="${p.pick ? 2 : 1}" data-ob="${esc(p.symbol)}" data-tip="${esc(p.symbol)}|${p.change_pct}|${p.score}|${p.pick ? "pick" : p.event ? "event risk" : p.contender ? "top 20" : ""}"/>`;
      });
    });
    const up = pts.filter((p) => p.change_pct > 0).length, dn = pts.filter((p) => p.change_pct < 0).length;
    el.innerHTML = `<div class="card"><div class="card-head"><h2>Market map</h2><span class="muted small">every F&amp;O stock at its move on the day, grouped by sector</span><span class="grow"></span>
        <span class="side-bull">${up} up</span><span class="side-bear">${dn} down</span></div>
      <svg class="svgchart ob-map-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="Every F&O stock's move on the day by sector">
        ${ticks.map((t) => `<line x1="${X(t)}" x2="${X(t)}" y1="${pad.t - 6}" y2="${h - pad.b}" stroke="${t === 0 ? "var(--axis)" : "var(--grid)"}"/><text x="${X(t)}" y="${pad.t - 10}" text-anchor="middle" font-size="11" fill="var(--muted)">${t > 0 ? "+" : ""}${t.toFixed(1)}%</text>`).join("")}
        ${secs.map((s, i) => { const q = pts.filter((p) => p.sector === s); return `<text x="${pad.l - 10}" y="${pad.t + i * rowH + rowH / 2 + 4}" text-anchor="end" font-size="11.5" fill="var(--ink-2)">${esc(s.length > 26 ? s.slice(0, 25) + "…" : s)}</text>
          <text x="${w - pad.r + 8}" y="${pad.t + i * rowH + rowH / 2 + 4}" font-size="11" fill="var(--muted)">${q.filter((p) => p.change_pct > 0).length}/${q.length} up</text>`; }).join("")}
        ${dots}</svg>
      <div class="chart-legend"><span><span class="swatch" style="background:var(--up)"></span>Up on the day</span><span><span class="swatch" style="background:var(--down)"></span>Down</span>
        <span><span class="swatch" style="background:var(--warning)"></span>Event risk (moved over 3 ATR or 8%)</span><span>Large dots: top 20 on a side · ringed: picks · moves beyond ±${lim}% sit at the edge</span></div></div>`;
    el.querySelectorAll("circle[data-tip]").forEach((c) => {
      c.addEventListener("mousemove", (e) => { const [s, ch, sc, tag] = c.dataset.tip.split("|"); showTip(e, `<b>${esc(s)}</b> ${pct(+ch, 2)}<br>strength ${(+sc).toFixed(2)}${tag ? "<br>" + esc(tag) : ""}`); });
      c.addEventListener("mouseleave", hideTip);
    });
  }
  function selectOB(sym) {
    obSel = sym; store.set("mv2.obsym", sym);
    $$("[data-ob]").forEach((el) => el.classList.toggle("selected", el.dataset.ob === sym));
    loadOBDetail(sym, true);
    if (obTab !== "board" && obTab !== "confirm") return;
    $("#ob-detail-card").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  async function loadOBDetail(sym, force) {
    if (!sym) return;
    if (!force && obDetail && obDetail.symbol === sym && Date.now() - obDetailAt < (obOpen() ? 30000 : 300000)) { drawOBDetail(); return; }
    await once("obd", async () => {
      try { obDetail = await api("/api/ob/stock/" + encodeURIComponent(sym)); } catch (e) { obDetail = { symbol: sym, error: e.message }; }
      obDetailAt = Date.now();
    });
    drawOBDetail();
  }
  function ensureOBChart() {
    if (obChart || !window.LightweightCharts) return;
    const el = $("#ob-chart");
    if (!el.clientWidth) return;
    obChart = LightweightCharts.createChart(el, Object.assign({ width: el.clientWidth, height: el.clientHeight, crosshair: { mode: 0 } }, chartTheme()));
    obCandles = obChart.addCandlestickSeries({});
    obVwap = obChart.addLineSeries({ lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    new ResizeObserver(() => { if (el.clientWidth) obChart.applyOptions({ width: el.clientWidth, height: el.clientHeight }); }).observe(el);
  }
  function drawOBDetail() {
    const d = obDetail;
    if (!d || page !== "ob") return;
    $("#ob-title").textContent = d.error ? `${d.symbol}: ${d.error}` : `${d.symbol}${d.name && d.name !== d.symbol ? " · " + d.name : ""}`;
    if (d.error) { $("#ob-fields").innerHTML = ""; return; }
    $("#ob-src").textContent = d.live ? "today, 5-minute candles · the last one live" : `last session ${fmtDay(d.session)}`;
    ensureOBChart();
    const r = d.row || {}, z = r.zone;
    if (obChart) {
      obChartLastT = d.bars.length ? d.bars[d.bars.length - 1].t : 0;
      const T = (t) => t + IST_SHIFT, up = cssVar("--up"), down = cssVar("--down"), muted = cssVar("--muted");
      obChart.applyOptions(chartTheme());
      obCandles.applyOptions({ upColor: "rgba(0,0,0,0)", borderUpColor: up, wickUpColor: up, downColor: down, borderDownColor: down, wickDownColor: down, borderVisible: true });
      obVwap.applyOptions({ color: cssVar("--series-1") });
      obCandles.setData(d.bars.map((b) => ({ time: T(b.t), open: b.o, high: b.h, low: b.l, close: b.c })));
      obVwap.setData(d.bars.map((b, i) => ({ time: T(b.t), value: d.vwap[i] })));
      obSeries.forEach((s) => obChart.removeSeries(s)); obSeries = [];
      obLines.forEach((l) => obCandles.removePriceLine(l)); obLines = [];
      const lastT = d.bars.length ? d.bars[d.bars.length - 1].t : 0;
      for (const zz of d.zones) {
        const cur = z && zz.formed_at === z.formed_at && zz.side === z.side;
        const col = zz.side > 0 ? up : down;
        for (const v of [zz.top, zz.bottom]) {
          const s = obChart.addLineSeries({ color: cur ? col : (zz.status === "broken" ? muted + "66" : col + "88"), lineWidth: cur ? 2 : 1, lineStyle: cur ? 0 : 2,
            priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
          const end = zz.broken_at ? zz.broken_at - 300 : Math.max(lastT, zz.ob_start + 300);
          s.setData([{ time: T(zz.ob_start), value: v }, { time: T(end), value: v }]);
          obSeries.push(s);
        }
      }
      const add = (price, color, style, title, width = 1) => { if (price != null) obLines.push(obCandles.createPriceLine({ price, color, lineWidth: width, lineStyle: style, axisLabelVisible: true, title })); };
      if (z) {
        add(z.z, cssVar("--ink"), 0, z.side > 0 ? "buy at Z" : "sell at Z", 1);
        add(z.invalid, cssVar("--critical"), 0, "stop-loss", 2);
        z.targets.forEach((t, i) => add(t, cssVar("--good"), 2, `T${i + 1}`, 1));
        if (!z.targets.length) add(zTarget(z), cssVar("--good"), 2, "2R", 1);
        (z.given_targets || []).forEach((t, i) => { if (!z.targets.includes(t)) add(t, cssVar("--good"), 1, `T${i + 1} as given`, 1); });
        z.behind.forEach((t) => add(t, cssVar("--warning"), 3, "", 1));
      }
      const marks = [];
      if (z && z.known_at) {
        const pos = z.side > 0 ? "belowBar" : "aboveBar";
        marks.push({ time: T(z.formed_at), position: pos, color: cssVar("--ink"), shape: "circle", text: `Z given ${fmtIST(z.known_at)}` });
        if (z.entry_at) marks.push({ time: T(z.entry_at), position: pos, color: z.side > 0 ? up : down, shape: z.side > 0 ? "arrowUp" : "arrowDown", text: `entry ${fmtIST(z.entry_at)}` });
      }
      obCandles.setMarkers(marks);
      add(d.prev_close, muted, 1, "prev close");
      if (d.cpr) { add(d.cpr.tc, cssVar("--cpr"), 2, "CPR top"); add(d.cpr.bc, cssVar("--cpr"), 2, "CPR bottom"); }
      if (obChartSym !== d.symbol) { obChart.timeScale().fitContent(); obChartSym = d.symbol; }
      obChartTail();
      obHeadLineSync();
    }
    drawOBFields();
  }
  // while a break is forming: the swing level it is breaking, on the chart (kept in step every second)
  let obHeadLine = null, obHeadLevel = null;
  function obHeadLineSync() {
    if (!obCandles) return;
    const d = obDetail, r = d && !d.error ? obRows().find((x) => x.symbol === d.symbol) : null, h = r && r.heads;
    const want = h && h.kind === "break" ? h.level : null;
    if (obHeadLine && want !== obHeadLevel) { try { obCandles.removePriceLine(obHeadLine); } catch (e) { /* chart reset */ } obHeadLine = null; }
    if (want != null && !obHeadLine) obHeadLine = obCandles.createPriceLine({ price: want, color: cssVar("--series-1"), lineWidth: 2, lineStyle: 2, axisLabelVisible: true, title: `swing ${r.side > 0 ? "high" : "low"}: breaking` });
    obHeadLevel = obHeadLine ? want : null;
  }
  // the candles that closed since the board's last refresh, and the forming one (from the live poll)
  function obChartTail() {
    const t = obLive && obLive.tail, d = obDetail;
    if (!obChart || !t || !d || t.symbol !== d.symbol) return;
    for (const b of t.bars) {
      if (b.t < obChartLastT) continue;
      try { obCandles.update({ time: b.t + IST_SHIFT, open: b.o, high: b.h, low: b.l, close: b.c }); obChartLastT = b.t; } catch (e) { /* older than the chart */ }
    }
  }
  function obDetailLive() {
    const d = obDetail;
    if (!d || d.error || page !== "ob") return;
    obChartTail();
    obHeadLineSync();
    drawOBFields();
  }
  function drawOBFields() {
    const d = obDetail;
    if (!d || d.error) return;
    const r = obRows().find((x) => x.symbol === d.symbol) || d.row || {}, z = r.zone;
    const c = r.checks || {};
    const be = z && z.rr != null ? Math.round(100 / (1 + z.rr)) : null;
    $("#ob-fields").innerHTML = `<div class="ob-fields"><dl>
      <dt>Side</dt><dd class="${sideCls(r.side)}">${sideWord(r.side)}${r.side ? ` · #${r.rank} of ${r.of}` : ""}</dd>
      <dt>Rank</dt><dd>${r.pctile != null ? `p${r.pctile} (stronger than ${r.pctile}% of F&amp;O stocks)` : "–"}</dd>
      <dt>Rule</dt><dd>${r.rule === "pick" ? `<span class="chip good">Pick</span>` : esc(r.rule || "–")}</dd>
      <dt>Price</dt><dd><b>₹${px(r.live_last != null ? r.live_last : r.last)}</b> · <span class="${pnlClass(r.change_pct)}">${pct(r.change_pct, 2)}</span>${r.live_last != null ? ` <span class="muted small">live</span>` : ""}</dd>
      <dt>Heads-up</dt><dd>${r.heads ? `${headChip(r)} ${esc(headText(r))}` : d.live ? `<span class="muted">none right now</span>` : "–"}</dd>
      <dt>Strength</dt><dd>${r.move_atr != null ? r.move_atr.toFixed(2) + " ATR" : "–"} · straight ${Math.round(100 * (r.clean || 0))}% <span class="muted small">(finished candles)</span></dd>
      <dt>Order block</dt><dd>${z ? `${px(z.bottom)} – ${px(z.top)} · ${esc(z.status)}${z.visits ? ` (${z.visits} visit${z.visits > 1 ? "s" : ""})` : ""}${z.strong ? " · strong push" : ""}` : (r.event ? "none: event risk" : "none yet")}</dd>
      <dt>${r.side < 0 ? "Sell at Z" : "Buy at Z"}</dt><dd>${z ? `${px(z.z)} · given at ${zGiven(z)}, when the ${fmtIST(z.formed_at)} candle closed through the last swing ${z.side > 0 ? "high" : "low"}` : "–"}</dd>
      <dt>Entry signal</dt><dd>${!z ? "–" : zEntry(z) ? `price came back to Z in the ${zEntry(z)}–${fmtIST(z.entry_at + 300)} candle (a limit order at Z fills there)`
        : d.live ? "not yet: waiting for price to come back to Z" : "price never came back to Z that session"}</dd>
      <dt>First stage</dt><dd>${d.live ? firstStage(r, z) : z ? `${zGiven(z)}–${fmtIST(z.known_at + firstWin())}, right after the zone was given` : "–"}</dd>
      <dt>Stop-loss</dt><dd>${z ? `<b class="neg">${px(z.invalid)}</b> · risk ${zRisk(z)} from Z (just past the zone: a 5-minute close beyond it ends the idea)` : "–"}</dd>
      <dt>Target as given</dt><dd>${z ? `<b class="pos">${zGivenTargets(z)}</b> · the levels when the zone was given (${zGiven(z)}); they stay${z.given_rr != null ? ` · R:R 1:${z.given_rr.toFixed(1)} from Z` : ""} · the first-stage trade aims for the last one` : "–"}</dd>
      <dt>Next levels now</dt><dd>${z ? `${zTargets(z)} · ahead of the latest close, so they move with the price${z.targets.length > 1 ? " · the retest trade aims for T2 of these when it enters" : ""}` : "–"}</dd>
      <dt>R:R</dt><dd>${z ? `${rr(z)}${be != null ? ` · needs ${be}% winners to break even before costs` : ""}` : "–"}</dd>
      <dt>Paper trades</dt><dd>${[paperLine(d.symbol), paperLine(d.symbol, "paper_first")].filter(Boolean).join("<br>") || "–"}</dd>
      <dt>OI yesterday</dt><dd>${r.oi_buildup ? `${esc(r.oi_buildup)} (${pct(r.oi_chg_pct, 1)} futures OI)` : "–"}</dd>
      <dt>Daily ATR</dt><dd>${px(r.atr)}</dd></dl>
      <div class="checks"><b>Alignment ${r.aligned ?? 0}/4</b>
        <span>${c.stock ? "✓" : "✗"} stock is in the top 20 on its side</span><span>${c.sector ? "✓" : "✗"} its sector moves the same way</span>
        <span>${c.breadth ? "✓" : "✗"} most F&amp;O stocks move the same way</span><span>${c.vwap ? "✓" : "✗"} price is on the right side of VWAP</span></div>
      <div class="checks"><b>Confirmations ${r.confirm ? `${r.confirm.score}/${r.confirm.of}` : "–"}</b>
        ${((ob && ob.confirm_rules) || []).map((x) => { const k = r.confirm && r.confirm.checks[x.key]; return `<span>${!k || k.ok == null ? "–" : k.ok ? "✓" : "✗"} ${esc(x.label)}: ${esc((k && k.text) || "no data")}</span>`; }).join("")}</div>
      <p class="note">4 of 4: full attention · 3 of 4: smaller size · 2 or fewer: usually a pass. Waiting for price to come back to the zone gives the tightest stop;
        the first stage enters right away, so its stop is wider and its R:R smaller. Neither the heads-up nor the first stage is backtested yet.</p></div>`;
  }
  function renderHomeOB() {
    const el = $("#home-ob");
    if (!el) return;
    if (!ob || obStale()) loadOB().then(() => { if (page === "home") renderHomeOB(); });
    if (!ob || !ob.ready) { el.innerHTML = `<p class="empty">${esc((ob && ob.error) || "Building the order-block board…")}</p>`; return; }
    $("#home-ob-note").textContent = ob.live ? `live · ${fmtIST(ob.updated_at)}` : `last session ${fmtDay(ob.session)}`;
    const picks = ob.picks || [];
    el.innerHTML = picks.length ? `<div class="table-wrap"><table><thead><tr><th>Stock</th><th>Side</th><th class="r">Change</th><th class="r">Entry Z</th><th class="r">Stop-loss</th><th class="r">Target</th><th class="r">R:R</th><th class="r">Entry signal</th><th>Aligned</th></tr></thead><tbody>${picks.map((r) =>
      `<tr class="clickable" data-ob-go="${esc(r.symbol)}"><td><b>${esc(r.symbol)}</b><span class="name">${esc(r.sector)}</span></td><td class="${sideCls(r.side)}">${sideWord(r.side)}</td>
      <td class="r ${pnlClass(r.change_pct)}">${pct(r.change_pct, 2)}</td><td class="r">${r.zone ? `${px(r.zone.z)}<span class="name">given ${zGiven(r.zone)}</span>` : "–"}</td>
      <td class="r neg">${r.zone ? px(r.zone.invalid) : "–"}</td><td class="r pos">${r.zone ? px(zTarget(r.zone)) : "–"}</td><td class="r">${rr(r.zone)}</td>
      <td class="r">${!r.zone ? "–" : zEntry(r.zone) || `<span class="muted">${ob.live ? "waiting" : "none"}</span>`}</td><td>${alignDots(r)}</td></tr>`).join("")}</tbody></table></div>` : `<p class="empty">No picks passed the rules.</p>`;
  }
  // ------------------------------------------------------------------ crypto & XAUUSD (Binance)
  let cx = null, cxAt = 0, cxSel = store.get("mv2.cxsym", "XAUUSDT"), cxD = null, cxDAt = 0;
  const cxPrev = {};                     // the price last shown per pair: a cell flashes when it changes
  const cxCharts = {};                   // "cx" (live prices) and "cxob" (order blocks) each keep a chart
  // a pair's price decimals: its Binance tick size, else by the size of the price
  const cdec = (v, s) => { const t = ((cx && cx.dec) || {})[s]; if (t != null) return Math.min(8, t); const a = Math.abs(v || 0); return a >= 100 ? 2 : a >= 1 ? 4 : a > 0 ? Math.min(8, Math.max(4, 3 - Math.floor(Math.log10(a)))) : 2; };
  const cpx = (v, s) => (v == null || isNaN(v) ? "–" : new Intl.NumberFormat("en-US", { minimumFractionDigits: cdec(v, s), maximumFractionDigits: cdec(v, s) }).format(v));
  const usd = (v) => (v == null ? "–" : "$" + (v >= 1e9 ? (v / 1e9).toFixed(2) + "B" : v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(0) + "k" : v.toFixed(0)));
  const cxShort = (s) => (s === "XAUUSDT" ? "XAUUSD" : String(s).replace(/USDT$/, ""));
  const cxLive = (s) => ((cx && cx.live) || {})[s] || {};
  const cxRows = () => (cx && cx.ready ? [...(cx.gold ? [cx.gold] : []), ...(cx.rows || [])] : []);
  const cxRow = (s) => cxRows().find((r) => r.symbol === s);
  const cxTargets = (z, s) => (z.targets.length ? z.targets.map((t, i) => `T${i + 1} ${cpx(t, s)}`).join(" · ") : `${cpx(zTarget(z), s)} (2R: no level ahead)`);
  const CXALIGN = { rank: "in the top 5 on its side", btc: "Bitcoin moves the same way today", breadth: "most coins move the same way", vwap: "on the right side of today's VWAP" };
  function cxDots(r) {
    const c = r.checks || {}, keys = Object.keys(CXALIGN).filter((k) => k in c);
    return `<span class="align" title="${esc(keys.map((k) => (c[k] == null ? "– " : c[k] ? "✓ " : "✗ ") + CXALIGN[k] + (c[k] == null ? " (not counted)" : "")).join("\n"))}">${keys.map((k) => `<i class="${c[k] ? "on" : ""}${c[k] == null ? " na" : ""}"></i>`).join("")}</span>`;
  }
  function cxSpark(pts) {
    if (!pts || pts.length < 2) return "";
    const lo = Math.min(...pts), hi = Math.max(...pts), span = hi - lo || 1, w = 200, h = 34;
    const d = pts.map((v, i) => `${i ? "L" : "M"}${(i * w / (pts.length - 1)).toFixed(1)},${(h - 2 - (h - 4) * (v - lo) / span).toFixed(1)}`).join("");
    return `<svg class="cx-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true"><path d="${d}" fill="none" stroke="${pts[pts.length - 1] >= pts[0] ? "var(--up)" : "var(--down)"}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>`;
  }
  function cxFlash() {
    const out = {};
    for (const [s, L] of Object.entries((cx && cx.live) || {})) {
      const was = cxPrev[s];
      out[s] = was == null || L.price === was ? "" : L.price > was ? "flash-up" : "flash-down";
      cxPrev[s] = L.price;
    }
    return out;
  }
  function cxLoad() {
    return once("cx", async () => {
      try { cx = await api("/api/cx/board"); } catch (e) { cx = { ready: false, error: e.message }; }
      cxAt = Date.now();
    });
  }
  async function cxLoadDetail(force) {
    if (!cxSel || (!force && cxD && cxD.symbol === cxSel && Date.now() - cxDAt < 1800)) return;
    await once("cxd", async () => {
      try { cxD = await api("/api/cx/stock/" + encodeURIComponent(cxSel)); } catch (e) { cxD = { symbol: cxSel, error: e.message }; }
      cxDAt = Date.now();
    });
  }
  function renderCX(entering) {
    const which = page;
    if (entering) drawCX(which);
    if (entering || Date.now() - cxAt > 1500) cxLoad().then(() => { if (page === which) drawCX(which); });
    if (which !== "cxcf") cxLoadDetail(entering).then(() => { if (page === which) drawCXDetail(which); });
  }
  function cxSelect(sym) {
    cxSel = sym; store.set("mv2.cxsym", sym);
    if (page === "cxcf") { location.hash = "#cxob"; return; }
    $$("[data-cx]").forEach((el) => el.classList.toggle("selected", el.dataset.cx === sym));
    cxLoadDetail(true).then(() => drawCXDetail(page));
    const card = $(page === "cx" ? "#cx-chart-card" : "#cxob-detail-card");
    if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function cxHead(el) {
    if (!cx) { el.innerHTML = `<div class="card"><p class="empty">Connecting to Binance…</p></div>`; return false; }
    const st = (cx.streams || []).map((s) => {
      const fresh = s.connected && s.last_msg && Date.now() / 1000 - s.last_msg < 30;
      return `<span class="chip ${fresh ? "good" : "bad"}" title="${esc(s.error || "")}">${esc(s.name)}: ${fresh ? "live" : "reconnecting"}</span>`;
    }).join(" ");
    if (!cx.ready) { el.innerHTML = `<div class="card"><p class="empty">${esc(cx.error || "Loading candles from Binance…")}</p>${st ? `<div class="ob-status">${st}</div>` : ""}</div>`; return false; }
    const tot = Math.max(1, cx.up + cx.down);
    el.innerHTML = `<div class="card"><div class="ob-status">
        <div>${st} <span class="muted small">Binance websocket · board updated ${fmtIST(cx.updated_at)} · today = ${esc(fmtDay(cx.day))}, UTC day from 05:30 IST</span></div>
        <div class="breadth" title="Coins up and down since yesterday's close (UTC)"><span class="side-bull">${cx.up} up</span>
          <div class="bar"><span style="width:${(100 * cx.up / tot).toFixed(1)}%;background:var(--up)"></span><span style="width:${(100 * cx.down / tot).toFixed(1)}%;background:var(--down)"></span></div>
          <span class="side-bear">${cx.down} down</span></div>
        ${cx.btc_chg != null ? `<div class="small">Bitcoin <span class="${pnlClass(cx.btc_chg)}">${pct(cx.btc_chg, 2)}</span> today</div>` : ""}
      </div>${cx.error ? `<p class="note" style="margin:0">${esc(cx.error)}</p>` : ""}</div>`;
    if (!cxRow(cxSel)) cxSel = (cx.gold || (cx.rows || [])[0] || {}).symbol || null;
    return true;
  }
  function drawCX(which) {
    const ok = cxHead($(`#${which}-head`));
    if (which === "cx") drawCXLive(ok);
    if (which === "cxob") drawCXBoard(ok);
    if (which === "cxcf") drawCXConfirm(ok);
  }
  function cxSideCell(r) {
    if (!r.side) return `<span class="muted small">${r.event ? "event risk" : "flat or choppy"}</span>`;
    return `<span class="${sideCls(r.side)}">${sideWord(r.side)}</span> <span class="small num">${(r.score || 0).toFixed(2)}</span>` +
      `<span class="name">${r.rank ? `#${r.rank} · ` : ""}${r.rule === "pick" ? "pick" : esc((r.rule || "").replace("not picked · ", ""))}</span>`;
  }
  // live prices
  function drawCXLive(ok) {
    $("#cx-note").innerHTML = `Prices and analytics only: the bot never trades crypto or gold. Crypto prices are Binance spot USDT pairs; XAUUSD is Binance's XAUUSDT perpetual,
      which tracks spot gold in US dollars an ounce. For Indian residents: trading XAUUSD or other forex CFDs with offshore brokers is not permitted under FEMA (the RBI keeps
      an alert list of such platforms); gold can be traded in India through MCX gold futures, gold ETFs or Sovereign Gold Bonds. Crypto gains in India are taxed at 30% with 1% TDS. Not investment advice.`;
    if (!ok) { $("#cx-cards").innerHTML = ""; $("#cx-table").innerHTML = ""; return; }
    const fl = cxFlash();
    $("#cx-cards").innerHTML = ["XAUUSDT", "BTCUSDT", "ETHUSDT"].map((s) => {
      const r = cxRow(s), L = cxLive(s);
      if (!r) return "";
      return `<div class="tile cx-tile ${s === cxSel ? "selected" : ""}" data-cx="${s}" tabindex="0">
        <div class="label">${esc(r.name)} <span class="muted">· ${cxShort(s)}</span></div>
        <div class="value num ${fl[s] || ""}">$${cpx(L.price, s)}</div>
        <div class="sub"><span class="${pnlClass(L.chg_day)}">${pct(L.chg_day, 2)} today</span> · <span class="${pnlClass(L.chg24)}">${pct(L.chg24, 2)} 24h</span> · 24h range ${cpx(L.low24, s)} – ${cpx(L.high24, s)}</div>
        ${cxSpark(r.spark)}</div>`;
    }).join("");
    $("#cx-table").innerHTML = `<div class="table-wrap"><table class="ob-table cx-table"><thead><tr><th>Pair</th><th class="r">Price (USD)</th><th class="r">Today</th><th class="r">24 hours</th>
      <th class="r">24h low – high</th><th class="r">24h volume</th><th>Board</th><th>Zone</th></tr></thead><tbody>${cxRows().map((r) => {
        const L = cxLive(r.symbol), z = r.zone, zc = z ? ZCHIP[z.status] : null;
        return `<tr class="clickable ${r.symbol === cxSel ? "selected" : ""}" data-cx="${r.symbol}" tabindex="0">
          <td><b>${cxShort(r.symbol)}</b><span class="name">${esc(r.name)} · ${esc(r.sector)}</span></td>
          <td class="r num ${fl[r.symbol] || ""}"><b>${cpx(L.price, r.symbol)}</b></td><td class="r ${pnlClass(L.chg_day)}">${pct(L.chg_day, 2)}</td><td class="r ${pnlClass(L.chg24)}">${pct(L.chg24, 2)}</td>
          <td class="r num small">${cpx(L.low24, r.symbol)} – ${cpx(L.high24, r.symbol)}</td><td class="r">${usd(L.qvol24)}</td><td>${cxSideCell(r)}</td>
          <td>${z ? `<span class="chip ${zc[0]}">${zc[1]}</span><span class="name num">Z ${cpx(z.z, r.symbol)} · SL ${cpx(z.invalid, r.symbol)}</span>` : `<span class="muted small">–</span>`}</td></tr>`;
      }).join("")}</tbody></table></div>`;
  }
  // charts
  function cxChart(key) {
    if (cxCharts[key]) return cxCharts[key];
    const el = $(`#${key}-chart`);
    if (!window.LightweightCharts || !el || !el.clientWidth) return null;
    const chart = LightweightCharts.createChart(el, Object.assign({ width: el.clientWidth, height: el.clientHeight, crosshair: { mode: 0 } }, chartTheme()));
    const c = cxCharts[key] = { chart, candles: chart.addCandlestickSeries({}), series: [], lines: [], sym: null, lastT: 0,
      vwap: chart.addLineSeries({ lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }) };
    new ResizeObserver(() => { if (el.clientWidth) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }); }).observe(el);
    return c;
  }
  function cxDrawChart(key, d, zones) {
    const c = cxChart(key);
    if (!c || !d || !d.bars || !d.bars.length) return;
    const T = (t) => t + IST_SHIFT, up = cssVar("--up"), down = cssVar("--down"), muted = cssVar("--muted");
    const p = cdec(d.bars[d.bars.length - 1].c, d.symbol), fmt = { type: "price", precision: p, minMove: Math.pow(10, -p) };
    const same = c.sym === d.symbol;
    c.chart.applyOptions(chartTheme());
    c.candles.applyOptions({ upColor: "rgba(0,0,0,0)", borderUpColor: up, wickUpColor: up, downColor: down, borderDownColor: down, wickDownColor: down, borderVisible: true, priceFormat: fmt });
    c.vwap.applyOptions({ color: cssVar("--series-1"), priceFormat: fmt });
    const cs = (b) => ({ time: T(b.t), open: b.o, high: b.h, low: b.l, close: b.c });
    if (same && c.lastT) d.bars.filter((b) => b.t >= c.lastT).forEach((b) => c.candles.update(cs(b)));
    else c.candles.setData(d.bars.map(cs));
    c.lastT = d.bars[d.bars.length - 1].t;
    c.vwap.setData(d.vwap.map((x) => ({ time: T(x.t), value: x.v })));
    c.series.forEach((s) => c.chart.removeSeries(s)); c.series = [];
    c.lines.forEach((l) => c.candles.removePriceLine(l)); c.lines = [];
    const add = (price, color, style, title, width = 1) => { if (price != null) c.lines.push(c.candles.createPriceLine({ price, color, lineWidth: width, lineStyle: style, axisLabelVisible: true, title })); };
    const z = (d.row || {}).zone, marks = [];
    if (zones) {
      for (const zz of d.zones.slice(-8)) {
        const cur = z && zz.formed_at === z.formed_at && zz.side === z.side, col = zz.side > 0 ? up : down;
        const end = zz.broken_at ? zz.broken_at - 300 : Math.max(c.lastT, zz.ob_start + 300);
        for (const v of [zz.top, zz.bottom]) {
          const s = c.chart.addLineSeries({ color: cur ? col : (zz.status === "broken" ? muted + "66" : col + "88"), lineWidth: cur ? 2 : 1, lineStyle: cur ? 0 : 2,
            priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, priceFormat: fmt });
          s.setData([{ time: T(zz.ob_start), value: v }, { time: T(end), value: v }]);
          c.series.push(s);
        }
      }
      if (z) {
        add(z.z, cssVar("--ink"), 0, z.side > 0 ? "buy at Z" : "sell at Z");
        add(z.invalid, cssVar("--critical"), 0, "stop-loss", 2);
        z.targets.forEach((t, i) => add(t, cssVar("--good"), 2, `T${i + 1}`));
        if (!z.targets.length) add(zTarget(z), cssVar("--good"), 2, "2R");
        const pos = z.side > 0 ? "belowBar" : "aboveBar";
        marks.push({ time: T(z.formed_at), position: pos, color: cssVar("--ink"), shape: "circle", text: `Z given ${fmtIST(z.known_at)}` });
        if (z.entry_at) marks.push({ time: T(z.entry_at), position: pos, color: z.side > 0 ? up : down, shape: z.side > 0 ? "arrowUp" : "arrowDown", text: `entry ${fmtIST(z.entry_at)}` });
      }
      if (d.cpr) { add(d.cpr.tc, cssVar("--cpr"), 2, "CPR"); add(d.cpr.bc, cssVar("--cpr"), 2, ""); }
    }
    add(d.prev_close, muted, 1, "prev close");
    c.candles.setMarkers(marks);
    if (!same) { c.chart.timeScale().fitContent(); c.sym = d.symbol; }
  }
  function drawCXDetail(which) {
    const d = cxD;
    if (!d || (which !== "cx" && which !== "cxob")) return;
    const title = d.error ? `${cxShort(d.symbol)}: ${d.error}` : `${cxShort(d.symbol)} · ${d.name}`;
    if (which === "cx") {
      $("#cx-title").textContent = title;
      $("#cx-src").textContent = d.error ? "" : `$${cpx((d.live || {}).price, d.symbol)} · ${d.gold ? "Binance XAUUSDT perpetual" : "Binance spot"}`;
      if (!d.error) cxDrawChart("cx", d, false);
      return;
    }
    $("#cxob-title").textContent = title;
    if (d.error) { $("#cxob-fields").innerHTML = ""; return; }
    $("#cxob-src").textContent = "finished 5-minute candles, last 24 hours; zones are today's (UTC day)";
    cxDrawChart("cxob", d, true);
    const r = d.row || {}, z = r.zone, L = d.live || {}, c = r.confirm, rules = (cx && cx.confirm_rules) || [];
    $("#cxob-fields").innerHTML = `<div class="ob-fields"><dl>
      <dt>Price</dt><dd><b>$${cpx(L.price, d.symbol)}</b> · <span class="${pnlClass(L.chg_day)}">${pct(L.chg_day, 2)} today</span> · ${pct(L.chg24, 2)} over 24 hours</dd>
      <dt>Side</dt><dd class="${sideCls(r.side)}">${sideWord(r.side)}${r.rank ? ` · #${r.rank} of ${r.of}` : ""}</dd>
      <dt>Rule</dt><dd>${r.rule === "pick" ? `<span class="chip good">Pick</span>` : esc(r.rule || "–")}</dd>
      <dt>Strength</dt><dd>${(r.score || 0).toFixed(2)} · ${r.move_atr != null ? r.move_atr.toFixed(2) + " ATR" : "–"} · straight ${Math.round(100 * (r.clean || 0))}%</dd>
      <dt>Order block</dt><dd>${z ? `${cpx(z.bottom, d.symbol)} – ${cpx(z.top, d.symbol)} · ${esc(z.status)}${z.visits ? ` (${z.visits} visit${z.visits > 1 ? "s" : ""})` : ""}${z.strong ? " · strong push" : ""}` : r.event ? "none: event risk" : r.side ? "none yet" : "none: the pair is flat or choppy today"}</dd>
      <dt>${r.side < 0 ? "Sell at Z" : "Buy at Z"}</dt><dd>${z ? `<b>${cpx(z.z, d.symbol)}</b> · given at ${zGiven(z)}, when the ${fmtIST(z.formed_at)} candle closed through the last swing ${z.side > 0 ? "high" : "low"}` : "–"}</dd>
      <dt>Entry signal</dt><dd>${!z ? "–" : zEntry(z) ? `price came back to Z in the ${zEntry(z)}–${fmtIST(z.entry_at + 300)} candle` : "not yet: waiting for price to come back to Z"}</dd>
      <dt>Stop-loss</dt><dd>${z ? `<b class="neg">${cpx(z.invalid, d.symbol)}</b> · risk ${cpx(Math.abs(z.z - z.invalid), d.symbol)} (${(100 * Math.abs(z.z - z.invalid) / z.z).toFixed(2)}%) from Z` : "–"}</dd>
      <dt>Target</dt><dd>${z ? `<b class="pos">${cxTargets(z, d.symbol)}</b>` : "–"}</dd>
      <dt>R:R</dt><dd>${rr(z)}</dd>
      <dt>Futures OI</dt><dd>${r.oi_chg != null ? `${pct(r.oi_chg, 1)} since 05:30 IST` : "–"}${r.funding != null ? ` · funding ${(100 * r.funding).toFixed(4)}% every 8 hours` : ""}</dd>
      <dt>Daily ATR</dt><dd>${cpx(r.atr, d.symbol)}</dd></dl>
      <div class="checks"><b>Alignment ${r.aligned ?? 0}/${r.aligned_of ?? 0}</b>${Object.keys(CXALIGN).filter((k) => r.checks && k in r.checks).map((k) =>
        `<span>${r.checks[k] == null ? "–" : r.checks[k] ? "✓" : "✗"} ${CXALIGN[k]}${r.checks[k] == null ? " (not counted)" : ""}</span>`).join("")}</div>
      <div class="checks"><b>Confirmations ${c ? `${c.score}/${c.of}` : "–"}</b>${c ? rules.map((x) => { const k = c.checks[x.key]; return `<span>${!k || k.ok == null ? "–" : k.ok ? "✓" : "✗"} ${esc(x.label)}: ${esc((k && k.text) || "no data")}</span>`; }).join("")
        : `<span class="muted">only for a pair on a side of the board</span>`}</div>
      <p class="note">Wait for price to come back to Z; chasing worsens R:R. Not backtested on crypto or gold.</p></div>`;
  }
  // order blocks
  function cxPickCard(r) {
    const z = r.zone, L = cxLive(r.symbol);
    return `<div class="pick ${r.symbol === cxSel ? "selected" : ""}" data-cx="${r.symbol}" tabindex="0">
      <div class="top"><b>${cxShort(r.symbol)}</b><span class="grow"></span><span class="${pnlClass(L.chg_day)} num">${pct(L.chg_day, 2)}</span></div>
      <div class="small"><span class="${sideCls(r.side)}">${sideWord(r.side)}</span> · <span class="num">$${cpx(L.price, r.symbol)}</span></div>
      <div class="muted small">${esc(r.sector)}${r.rank ? ` · #${r.rank}` : ""} · ${r.aligned}/${r.aligned_of} aligned${r.confirm ? ` · ${r.confirm.score}/${r.confirm.of} confirmed` : ""}</div>
      <div class="lv">${z ? `<span>${r.side < 0 ? "Sell zone" : "Buy zone"}</span><span>${cpx(z.bottom, r.symbol)} – ${cpx(z.top, r.symbol)} (${esc(z.status)})</span>
        <span>${r.side < 0 ? "Sell at Z" : "Buy at Z"}</span><span><b>${cpx(z.z, r.symbol)}</b> · given ${zGiven(z)}</span>
        <span>Stop-loss</span><span><b class="neg">${cpx(z.invalid, r.symbol)}</b></span>
        <span>Target</span><span><b class="pos">${cxTargets(z, r.symbol)}</b></span><span>R:R</span><span>${rr(z)}</span>
        <span>Entry signal</span><span>${zEntry(z) ? `back at Z ${zEntry(z)}` : "waiting for price to return"}</span>`
        : `<span>Zone</span><span>${r.side ? "no order block yet" : r.event ? "event risk: no levels" : "flat or choppy today"}</span>`}</div></div>`;
  }
  function cxBoardTable(rows, side) {
    if (!rows.length) return `<p class="empty">No ${side > 0 ? "bullish" : "bearish"} coin is moving cleanly right now.</p>`;
    const mx = Math.max(...rows.map((r) => Math.abs(r.score || 0)), 0.01);
    return `<div class="table-wrap"><table class="ob-table"><thead><tr><th>#</th><th>Pair</th><th class="r">Today</th><th>Strength</th><th>Entry · stop-loss · target</th><th>Status</th></tr></thead><tbody>${rows.map((r) => {
      const z = r.zone, zc = z ? ZCHIP[z.status] : null, L = cxLive(r.symbol);
      const rule = r.rule === "pick" ? `<span class="chip good">Pick</span>` : r.event ? `<span class="chip warn">Event risk</span>` : `<span class="muted small">${esc((r.rule || "").replace("not picked · ", ""))}</span>`;
      return `<tr class="clickable ${r.symbol === cxSel ? "selected" : ""}" data-cx="${r.symbol}" tabindex="0">
        <td class="num">${r.rank}</td><td><b>${cxShort(r.symbol)}</b><span class="name">${esc(r.sector)} · $${cpx(L.price, r.symbol)}</span></td>
        <td class="r ${pnlClass(L.chg_day)}">${pct(L.chg_day, 2)}</td>
        <td class="num"><span class="sbar" style="width:${(42 * Math.abs(r.score) / mx).toFixed(0)}px;background:${side > 0 ? "var(--up)" : "var(--down)"}"></span>${(r.score || 0).toFixed(2)}<span class="name">straight ${Math.round(100 * (r.clean || 0))}%</span></td>
        <td>${z ? `<span class="chip ${zc[0]}">${zc[1]}</span><span class="name num">Z ${cpx(z.z, r.symbol)} · ${rr(z)}</span><span class="name num">SL <span class="neg">${cpx(z.invalid, r.symbol)}</span> · T <span class="pos">${cpx(zTarget(z), r.symbol)}</span></span><span class="name num">${zWhen(z)}</span>`
          : `<span class="muted small">${r.event ? "no levels" : "none yet"}</span>`}</td>
        <td>${rule}<span class="name">${cxDots(r)} ${r.aligned}/${r.aligned_of}${r.confirm ? ` · ${r.confirm.score}/${r.confirm.of} confirmed` : ""}</span></td></tr>`;
    }).join("")}</tbody></table></div>`;
  }
  function cxChoppy() {
    const flat = (cx.rows || []).filter((r) => !r.side);
    return flat.length ? `<p class="note">Flat or choppy today, so not on a side: ${flat.map((r) => `<a href="#cxob" data-cx="${r.symbol}">${cxShort(r.symbol)}</a>`).join(" · ")}
      (straightness under 25%: today's path went back and forth more than it went anywhere).</p>` : "";
  }
  function drawCXBoard(ok) {
    if (!ok) { $("#cxob-gold").innerHTML = ""; $("#cxob-board").innerHTML = ""; return; }
    const g = cx.gold;
    $("#cxob-gold").innerHTML = g ? `<div class="card cx-gold"><div class="card-head"><h2>XAUUSD · gold</h2><span class="muted small">Binance XAUUSDT perpetual · shown on its own, not ranked with coins</span></div>
      <div class="ob-picks" style="margin:0">${cxPickCard(g)}</div></div>` : "";
    const picks = cx.picks || [];
    $("#cxob-board").innerHTML = `<h2 class="cx-h">Crypto picks</h2>${picks.length ? `<div class="ob-picks">${picks.map(cxPickCard).join("")}</div>` : `<div class="card"><p class="empty">No picks: no coin passed the rules right now.</p></div>`}
      <div class="ob-boards">
        <div class="card flush"><div class="card-head"><h2 class="side-bull">Bullish board</h2><span class="muted small">by strength today · for buyers</span></div>${cxBoardTable(cx.bullish || [], 1)}</div>
        <div class="card flush"><div class="card-head"><h2 class="side-bear">Bearish board</h2><span class="muted small">by strength today · for sellers</span></div>${cxBoardTable(cx.bearish || [], -1)}</div>
      </div>${cxChoppy()}`;
  }
  // confirmations
  function cxCfTable(rows, label) {
    const rules = cx.confirm_rules || [];
    rows = rows.filter((r) => r.confirm).sort((a, b) => b.confirm.score - a.confirm.score || (a.rank || 0) - (b.rank || 0));
    if (!rows.length) return `<p class="empty">${esc(label)}</p>`;
    return `<div class="table-wrap"><table class="ob-table cf-table"><thead><tr><th>Pair</th><th class="r">Today</th><th>Confirmed</th>
      ${rules.map((x) => `<th class="cf" title="${esc(x.rule)}">${esc(x.label)}</th>`).join("")}<th>Board</th><th>Zone</th></tr></thead><tbody>${rows.map((r) => {
        const c = r.confirm, z = r.zone, zc = z ? ZCHIP[z.status] : null, L = cxLive(r.symbol);
        return `<tr class="clickable" data-cx="${r.symbol}" tabindex="0">
          <td><b>${cxShort(r.symbol)}</b><span class="name">${r.rank ? `#${r.rank} · ` : ""}<span class="${sideCls(r.side)}">${sideWord(r.side)}</span> · ${esc(r.sector)}</span></td>
          <td class="r ${pnlClass(L.chg_day)}">${pct(L.chg_day, 2)}</td>
          <td class="num"><span class="sbar" style="width:${(42 * c.score / Math.max(1, rules.length)).toFixed(0)}px;background:${r.side > 0 ? "var(--up)" : "var(--down)"}"></span>${c.score}/${c.of}</td>
          ${rules.map((x) => { const k = c.checks[x.key]; return `<td class="cf" title="${esc(x.label + ": " + ((k && k.text) || "no data"))}">${cfMark(k)}<span class="name num">${esc((k && k.v) || "–")}</span></td>`; }).join("")}
          <td>${cxDots(r)} <span class="small">${r.aligned}/${r.aligned_of}</span>${r.rule === "pick" ? ` <span class="chip good">Pick</span>` : ""}</td>
          <td>${z ? `<span class="chip ${zc[0]}">${zc[1]}</span><span class="name num">Z ${cpx(z.z, r.symbol)}</span>` : `<span class="muted small">none yet</span>`}</td></tr>`;
      }).join("")}</tbody></table></div>`;
  }
  function drawCXConfirm(ok) {
    if (!ok) { $("#cxcf-body").innerHTML = ""; return; }
    const g = cx.gold;
    $("#cxcf-body").innerHTML = `<div class="card"><p class="note" style="margin:0">Each check asks whether that piece of evidence agrees with the pair's side on the board: ✓ agrees, ✗ disagrees, – no data (not counted).
        Hover a column for its rule, a cell for its numbers; click a row for its chart. Futures open interest and funding come from Binance's USD-M perpetuals, re-read every 5 minutes
        ${cx.slow_at ? `(last ${fmtIST(cx.slow_at)})` : ""}; the rest from Binance candles. Not part of the picks, and not backtested.</p></div>
      <div class="card flush"><div class="card-head"><h2>XAUUSD · gold</h2><span class="muted small">its own side, not ranked with coins</span></div>${cxCfTable(g ? [g] : [], "Gold is flat or choppy today: no side to confirm.")}</div>
      <div class="card flush"><div class="card-head"><h2 class="side-bull">Bullish confirmations</h2><span class="muted small">coins on the bullish board</span></div>${cxCfTable(cx.bullish || [], "No bullish coin is moving cleanly right now.")}</div>
      <div class="card flush"><div class="card-head"><h2 class="side-bear">Bearish confirmations</h2><span class="muted small">coins on the bearish board</span></div>${cxCfTable(cx.bearish || [], "No bearish coin is moving cleanly right now.")}</div>
      ${cxChoppy()}`;
  }
  function calHeat(el, days) {
    if (!el) return;
    const byDate = new Map(days.map((d) => [d.date, d]));
    const end = new Date(), start = new Date(end.getTime() - 90 * 86400000);
    while (start.getDay() !== 1) start.setDate(start.getDate() - 1);
    const mx = Math.max(1, ...days.map((d) => Math.abs(d.net)));
    const cells = [];
    for (const d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
      if (d.getDay() === 0 || d.getDay() === 6) continue;
      const iso = d.toISOString().slice(0, 10), x = byDate.get(iso);
      const a = x ? (0.25 + 0.75 * Math.min(1, Math.abs(x.net) / mx)).toFixed(2) : 0;
      const bg = x ? `color-mix(in srgb, ${x.net >= 0 ? "var(--good)" : "var(--critical)"} ${Math.round(a * 100)}%, var(--surface-3))` : "";
      cells.push(`<i style="${bg ? "background:" + bg : ""}" data-tip="${iso}|${x ? x.net : ""}|${x ? x.trades : 0}|${x ? x.wins : 0}"></i>`);
    }
    el.innerHTML = `<div class="cal-wrap"><div class="cal-days"><span>Mon</span><span>Tue</span><span>Wed</span><span>Thu</span><span>Fri</span></div><div class="cal">${cells.join("")}</div></div>
      <p class="note">Each square is a trading day of the last 90: green made money, red lost; darker means bigger. Grey: no closed trades.</p>`;
    el.querySelectorAll("[data-tip]").forEach((c) => {
      c.addEventListener("mousemove", (e) => { const [dd, net, n, w] = c.dataset.tip.split("|"); showTip(e, `${esc(fmtDay(dd))}<br>${net !== "" ? `net ${signed(+net)}<br>${n} trades, ${w} winners` : "no closed trades"}`); });
      c.addEventListener("mouseleave", hideTip);
    });
  }

  // ------------------------------------------------------------------ go
  if (selected) $("#chart-symbol").value = selected;
  if (resultsSym) $("#results-form").symbol.value = resultsSym;
  setPage(page);
  refresh();
})();
