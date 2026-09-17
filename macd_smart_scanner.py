import time
import threading
import concurrent.futures
import requests
import pandas as pd
import datetime

# ============================================================
# SETTINGS
# ============================================================
MARKET_TYPE = "futures"
SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"
BASE_URL = FUTURES_BASE_URL if MARKET_TYPE == "futures" else SPOT_BASE_URL
FUTURES_CONTRACT_TYPE = "PERPETUAL"

# --- Telegram Alerts -------------------------------------------------
# Sends a message to your phone the moment a qualifying signal is found,
# plus a short summary at the end of each scan.
#
# Values are read from environment variables so the token never sits in
# plain text in this file — important since this file may end up in a
# PUBLIC GitHub repo (use GitHub Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
#
# Running locally on your own computer only (not uploading to GitHub)?
# You can instead just paste your token/chat id directly as the second
# argument of each os.environ.get(...) call below.
import os
TELEGRAM_ENABLED = True
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
# -----------------------------------------------------------------------

TIMEFRAME = "4h"

# MACD
FAST_LENGTH = 12
SLOW_LENGTH = 26
SIGNAL_LENGTH = 9

# Signal freshness
LOOKBACK_CANDLES = 3

# Volume
VOLUME_AVG_PERIOD = 20
VOLUME_MULTIPLIER = 2.0

# Price-action filters
MIN_BODY_RATIO = 0.45       # body / full range
MAX_OPPOSITE_WICK_RATIO = 0.35
BREAKOUT_LOOKBACK = 10      # previous highs/lows, excluding current candle

# Trend filter
TREND_EMA = 50

# Score
STRONG_SCORE = 5
MEDIUM_SCORE = 5

# --- STRICT MODE -------------------------------------------------
# When True, a signal is only reported if EVERY quality check passes
# (score == 6/6): volume spike + clean candle + trend aligned + breakout
# confirmed + the candle right after the cross also confirmed direction.
# This is the tightest the current filter set can go. It will NOT
# guarantee any specific win rate — see BACKTEST_MODE below for how to
# actually measure the win rate this produces on real historical data.
STRICT_MODE = True
STRICT_MIN_SCORE = 6
# -------------------------------------------------------------------

# Scan
CANDLE_LIMIT = 180
SCAN_ALL_QUOTES = True
QUOTE_ASSET = "USDT"
MAX_WORKERS = 15
REQUEST_DELAY = 0.05

SAVE_HTML_REPORT = True
HTML_REPORT_PATH = "macd_volume_smart_scan.html"
DEBUG_DUMP_SYMBOL_LIST = True

# --- BACKTEST MODE ---------------------------------------------------
# Instead of scanning the live market, replay history and measure how
# often a signal that passed these exact filters was actually followed
# by a real move in the predicted direction. This is the only honest
# way to know a real win rate for this setup (and even then, past
# performance does not guarantee future results).
BACKTEST_MODE = False              # set True to run a backtest instead of a live scan
BACKTEST_CANDLES = 1000            # how much history to pull per symbol
BACKTEST_FOLLOWTHROUGH = 5         # candles forward to check after the signal
BACKTEST_MIN_MOVE_PCT = 1.0        # price must move at least this % in the
                                     # predicted direction to count as a "win"
BACKTEST_SYMBOL_LIMIT = 40         # cap symbols scanned in backtest (speed);
                                     # set to None to backtest every symbol
# -----------------------------------------------------------------------

_print_lock = threading.Lock()


def send_telegram_message(text):
    """
    Send a message to your Telegram chat. Fails silently (prints a warning)
    so a Telegram hiccup never crashes the actual scan.
    """
    if not TELEGRAM_ENABLED:
        return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            with _print_lock:
                print(f"  (Telegram) Failed to send alert: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        with _print_lock:
            print(f"  (Telegram) Failed to send alert: {e}")


# ============================================================
# BINANCE DATA
# ============================================================
def get_tradable_symbols():
    if MARKET_TYPE == "futures":
        info_url = f"{BASE_URL}/fapi/v1/exchangeInfo"
        ticker_url = f"{BASE_URL}/fapi/v1/ticker/24hr"
    else:
        info_url = f"{BASE_URL}/api/v3/exchangeInfo"
        ticker_url = f"{BASE_URL}/api/v3/ticker/24hr"

    resp = requests.get(info_url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    listed = set()

    for s in data["symbols"]:
        if s["status"] != "TRADING":
            continue
        if not SCAN_ALL_QUOTES and s["quoteAsset"] != QUOTE_ASSET:
            continue

        if MARKET_TYPE == "futures":
            if s.get("contractType") != FUTURES_CONTRACT_TYPE:
                continue
        else:
            if not s.get("isSpotTradingAllowed"):
                continue
            perms = set(s.get("permissions", []))
            for group in s.get("permissionSets", []):
                perms.update(group)
            if perms and "SPOT" not in perms:
                continue

            base = s["baseAsset"]
            if base.endswith(("UP", "DOWN", "3L", "3S", "5L", "5S")):
                continue

        listed.add(s["symbol"])

    resp = requests.get(ticker_url, timeout=15)
    resp.raise_for_status()
    tickers = resp.json()

    active_symbols = []
    for t in tickers:
        sym = t["symbol"]
        if sym in listed and float(t.get("volume", 0)) > 0:
            active_symbols.append(sym)

    active_symbols = sorted(active_symbols)

    if DEBUG_DUMP_SYMBOL_LIST:
        with open("scanned_symbols.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(active_symbols))
        print(f"(Debug) scanned_symbols.txt saved — {len(active_symbols)} symbols")

    return active_symbols


def get_klines(symbol, interval, limit):
    path = "/fapi/v1/klines" if MARKET_TYPE == "futures" else "/api/v3/klines"
    url = f"{BASE_URL}{path}"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base_vol", "taker_quote_vol", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df


# ============================================================
# INDICATORS
# ============================================================
def calculate_macd(df, fast=FAST_LENGTH, slow=SLOW_LENGTH, signal=SIGNAL_LENGTH):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea


def add_indicators(df):
    df = df.copy()
    df["ema50"] = df["close"].ewm(span=TREND_EMA, adjust=False).mean()
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    df["body_ratio"] = df["body"] / df["range"].replace(0, pd.NA)

    # Candle wick sizes
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_wick_ratio"] = df["upper_wick"] / df["range"].replace(0, pd.NA)
    df["lower_wick_ratio"] = df["lower_wick"] / df["range"].replace(0, pd.NA)

    return df


# ============================================================
# SIGNAL LOGIC
# ============================================================
def find_recent_crossover(dif, dea, lookback=LOOKBACK_CANDLES):
    diff = dif - dea
    start = max(1, len(diff) - lookback)

    # Search newest -> oldest, so the returned cross is the most recent one.
    for i in range(len(diff) - 1, start - 1, -1):
        prev_val = diff.iloc[i - 1]
        curr_val = diff.iloc[i]

        if prev_val < 0 and curr_val > 0:
            return "bullish", i
        if prev_val > 0 and curr_val < 0:
            return "bearish", i

    return None, None


def is_crossover_at(dif, dea, idx):
    """Check whether a MACD crossover happens exactly at index idx (used for backtesting)."""
    if idx < 1:
        return None
    diff = dif - dea
    prev_val = diff.iloc[idx - 1]
    curr_val = diff.iloc[idx]
    if prev_val < 0 and curr_val > 0:
        return "bullish"
    if prev_val > 0 and curr_val < 0:
        return "bearish"
    return None


def volume_ratio(df, idx):
    if idx < VOLUME_AVG_PERIOD:
        return 0.0

    avg_volume = df["volume"].iloc[idx - VOLUME_AVG_PERIOD:idx].mean()
    if avg_volume <= 0:
        return 0.0

    return float(df["volume"].iloc[idx] / avg_volume)


def candle_quality(df, idx, direction):
    row = df.iloc[idx]

    body_ratio = float(row["body_ratio"]) if pd.notna(row["body_ratio"]) else 0.0
    upper = float(row["upper_wick_ratio"]) if pd.notna(row["upper_wick_ratio"]) else 1.0
    lower = float(row["lower_wick_ratio"]) if pd.notna(row["lower_wick_ratio"]) else 1.0

    if direction == "bullish":
        good_direction = row["close"] > row["open"]
        good_body = body_ratio >= MIN_BODY_RATIO
        good_wick = upper <= MAX_OPPOSITE_WICK_RATIO
    else:
        good_direction = row["close"] < row["open"]
        good_body = body_ratio >= MIN_BODY_RATIO
        good_wick = lower <= MAX_OPPOSITE_WICK_RATIO

    return good_direction, good_body, good_wick


def breakout_status(df, idx, direction):
    if idx < BREAKOUT_LOOKBACK:
        return False

    current = df.iloc[idx]
    prior = df.iloc[idx - BREAKOUT_LOOKBACK:idx]

    if direction == "bullish":
        level = prior["high"].max()
        return bool(current["close"] > level)

    level = prior["low"].min()
    return bool(current["close"] < level)


def trend_status(df, idx, direction):
    row = df.iloc[idx]

    if direction == "bullish":
        return bool(row["close"] > row["ema50"])
    return bool(row["close"] < row["ema50"])


def next_candle_confirmation(df, cross_idx, direction):
    # If crossover happened on the latest closed candle, no next candle exists yet.
    if cross_idx + 1 >= len(df):
        return None

    cross_close = df["close"].iloc[cross_idx]
    next_row = df.iloc[cross_idx + 1]

    if direction == "bullish":
        return bool(next_row["close"] > cross_close)

    return bool(next_row["close"] < cross_close)


def score_signal(df, direction, idx):
    vr = volume_ratio(df, idx)
    volume_ok = vr >= VOLUME_MULTIPLIER

    candle_dir, candle_body, candle_wick = candle_quality(df, idx, direction)
    trend_ok = trend_status(df, idx, direction)
    breakout_ok = breakout_status(df, idx, direction)
    next_confirm = next_candle_confirmation(df, idx, direction)

    # Score:
    # 1 MACD crossover (always true here)
    # 1 strong volume
    # 1 candle direction/body/wick quality
    # 1 trend
    # 1 breakout
    # 1 next candle confirmation, when available
    score = 1

    if volume_ok:
        score += 1
    if candle_dir and candle_body and candle_wick:
        score += 1
    if trend_ok:
        score += 1
    if breakout_ok:
        score += 1
    if next_confirm is True:
        score += 1

    # Strong = 5+; Medium = 4; Weak = <=3
    if score >= STRONG_SCORE:
        grade = "STRONG BUY" if direction == "bullish" else "STRONG SELL"
    elif score >= MEDIUM_SCORE:
        grade = "MEDIUM BUY" if direction == "bullish" else "MEDIUM SELL"
    else:
        grade = "WEAK BUY" if direction == "bullish" else "WEAK SELL"

    return {
        "score": score,
        "grade": grade,
        "vol_ratio": vr,
        "volume_ok": volume_ok,
        "candle_ok": candle_dir and candle_body and candle_wick,
        "trend_ok": trend_ok,
        "breakout_ok": breakout_ok,
        "next_confirm": next_confirm,
    }


def analyze_symbol(df):
    if len(df) < max(SLOW_LENGTH + SIGNAL_LENGTH + VOLUME_AVG_PERIOD,
                     TREND_EMA + 5, BREAKOUT_LOOKBACK + 5):
        return None

    # IMPORTANT: Binance kline endpoint normally returns the current/open candle
    # as the final row. We exclude it so the scanner works only on closed candles.
    df = df.iloc[:-1].copy()
    df = add_indicators(df)

    dif, dea = calculate_macd(df)
    direction, cross_idx = find_recent_crossover(dif, dea)

    if direction is None:
        return None

    result = score_signal(df, direction, cross_idx)

    if STRICT_MODE and result["score"] < STRICT_MIN_SCORE:
        return None

    result["direction"] = direction
    result["cross_idx"] = cross_idx
    result["price"] = df["close"].iloc[-1]

    return result


# ============================================================
# SCAN (live)
# ============================================================
def process_symbol(symbol, idx, total):
    time.sleep(REQUEST_DELAY)

    for attempt in range(3):
        try:
            df = get_klines(symbol, TIMEFRAME, CANDLE_LIMIT)
            break
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (429, 418):
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
        except Exception:
            return None
    else:
        return None

    try:
        result = analyze_symbol(df)
    except Exception:
        return None

    if result is None:
        return None

    result["symbol"] = symbol

    with _print_lock:
        print(
            f"{symbol}: {result['grade']} | "
            f"Score {result['score']}/6 | "
            f"Vol x{result['vol_ratio']:.1f} | "
            f"Trend {'OK' if result['trend_ok'] else 'NO'} | "
            f"Breakout {'OK' if result['breakout_ok'] else 'NO'}"
        )

    emoji = "\U0001F7E2" if result["direction"] == "bullish" else "\U0001F534"
    alert_text = (
        f"{emoji} {result['grade']}\n"
        f"Symbol: {symbol}\n"
        f"Timeframe: {TIMEFRAME}\n"
        f"Score: {result['score']}/6\n"
        f"Volume: x{result['vol_ratio']:.1f} avg\n"
        f"Trend aligned: {'Yes' if result['trend_ok'] else 'No'}\n"
        f"Breakout: {'Yes' if result['breakout_ok'] else 'No'}\n"
        f"Price: {result['price']:.6g}"
    )
    send_telegram_message(alert_text)

    return result


def print_table(rows):
    if not rows:
        print("  (koi result nahi mila)")
        return

    headers = [
        "Symbol", "Signal", "Score", "Vol xAvg",
        "Trend", "Breakout", "Next", "Price"
    ]

    data = []
    for r in rows:
        next_s = "YES" if r["next_confirm"] is True else (
            "NO" if r["next_confirm"] is False else "-"
        )
        data.append([
            r["symbol"],
            r["grade"],
            f"{r['score']}/6",
            f"x{r['vol_ratio']:.1f}",
            "YES" if r["trend_ok"] else "NO",
            "YES" if r["breakout_ok"] else "NO",
            next_s,
            f"{r['price']:.6g}"
        ])

    widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))


def save_html_report(rows):
    rows = sorted(rows, key=lambda r: (r["score"], r["vol_ratio"]), reverse=True)

    html_rows = ""
    for r in rows:
        html_rows += f"""
        <tr>
          <td>{r['symbol']}</td>
          <td>{r['grade']}</td>
          <td>{r['score']}/6</td>
          <td>x{r['vol_ratio']:.1f}</td>
          <td>{"YES" if r['trend_ok'] else "NO"}</td>
          <td>{"YES" if r['breakout_ok'] else "NO"}</td>
          <td>{"YES" if r['next_confirm'] is True else ("NO" if r['next_confirm'] is False else "-")}</td>
          <td>{r['price']:.6g}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>MACD + Volume Smart Scan</title>
<style>
body {{ font-family: Arial, sans-serif; background:#111; color:#eee; padding:24px; }}
table {{ border-collapse:collapse; width:100%; max-width:1000px; }}
th,td {{ padding:9px 12px; border-bottom:1px solid #333; text-align:left; }}
th {{ background:#222; }}
h1 {{ font-size:22px; }}
.note {{ max-width:900px; line-height:1.5; color:#bbb; }}
</style>
</head>
<body>
<h1>MACD + Volume Smart Scan</h1>
<p class="note">
Timeframe: {TIMEFRAME} |
MACD {FAST_LENGTH}/{SLOW_LENGTH}/{SIGNAL_LENGTH} |
Volume >= {VOLUME_MULTIPLIER}x previous {VOLUME_AVG_PERIOD}-candle average |
Trend EMA{TREND_EMA} |
Breakout lookback {BREAKOUT_LOOKBACK} |
Strict mode: {"ON (score must be 6/6)" if STRICT_MODE else "OFF"}
</p>
<table>
<tr>
<th>Symbol</th><th>Signal</th><th>Score</th><th>Vol</th>
<th>Trend</th><th>Breakout</th><th>Next Confirm</th><th>Price</th>
</tr>
{html_rows}
</table>
<p class="note">
IMPORTANT: A score is a probability filter, not a guarantee. No combination of
indicators can promise a fixed win rate — see the backtest report for the
actual measured historical performance of this exact setup.
</p>
</body>
</html>"""

    with open(HTML_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)


def scan_market():
    print(f"Fetching active {MARKET_TYPE.upper()} symbols from Binance...")
    symbols = get_tradable_symbols()

    print(
        f"Found {len(symbols)} symbols. "
        f"Scanning {TIMEFRAME} using CLOSED candles only "
        f"{'(STRICT MODE — score must be 6/6)' if STRICT_MODE else ''}...\n"
    )

    rows = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_symbol, sym, i, len(symbols)): sym
            for i, sym in enumerate(symbols, 1)
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                rows.append(result)

    bullish = [r for r in rows if r["direction"] == "bullish"]
    bearish = [r for r in rows if r["direction"] == "bearish"]

    bullish.sort(key=lambda r: (r["score"], r["vol_ratio"]), reverse=True)
    bearish.sort(key=lambda r: (r["score"], r["vol_ratio"]), reverse=True)

    print("\n" + "=" * 95)
    print("SMART SCAN COMPLETE")
    print("=" * 95)

    print("\nBULLISH SIGNALS:")
    print_table(bullish)

    print("\nBEARISH SIGNALS:")
    print_table(bearish)

    if SAVE_HTML_REPORT:
        save_html_report(rows)
        print(f"\nHTML report saved: {HTML_REPORT_PATH}")

    if TELEGRAM_ENABLED and rows:
        send_telegram_message(
            f"Scan complete ({TIMEFRAME}): {len(bullish)} bullish, "
            f"{len(bearish)} bearish signal(s) found."
        )


# ============================================================
# BACKTEST — measure the REAL historical win rate of this setup
# ============================================================
def backtest_symbol(symbol):
    """
    Replay a symbol's history: find every past MACD crossover, apply the
    exact same STRICT_MODE filter used in the live scanner, then check
    whether price actually moved BACKTEST_MIN_MOVE_PCT% in the predicted
    direction within BACKTEST_FOLLOWTHROUGH candles. Returns a list of
    trade result dicts.
    """
    try:
        raw = get_klines(symbol, TIMEFRAME, BACKTEST_CANDLES)
    except Exception:
        return []

    if len(raw) < max(SLOW_LENGTH + SIGNAL_LENGTH + VOLUME_AVG_PERIOD,
                       TREND_EMA + 5, BREAKOUT_LOOKBACK + 5) + BACKTEST_FOLLOWTHROUGH + 5:
        return []

    df = raw.iloc[:-1].copy()  # drop the still-open candle, same as live scan
    df = add_indicators(df)
    dif, dea = calculate_macd(df)

    trades = []
    min_idx = max(SLOW_LENGTH + SIGNAL_LENGTH, TREND_EMA, BREAKOUT_LOOKBACK, VOLUME_AVG_PERIOD)
    max_idx = len(df) - BACKTEST_FOLLOWTHROUGH - 1  # need future candles to grade the trade

    for idx in range(min_idx, max_idx):
        direction = is_crossover_at(dif, dea, idx)
        if direction is None:
            continue

        result = score_signal(df, direction, idx)
        if STRICT_MODE and result["score"] < STRICT_MIN_SCORE:
            continue

        entry_price = df["close"].iloc[idx]
        future_price = df["close"].iloc[idx + BACKTEST_FOLLOWTHROUGH]
        move_pct = (future_price - entry_price) / entry_price * 100

        if direction == "bullish":
            win = move_pct >= BACKTEST_MIN_MOVE_PCT
        else:
            win = move_pct <= -BACKTEST_MIN_MOVE_PCT

        trades.append({
            "symbol": symbol,
            "direction": direction,
            "score": result["score"],
            "move_pct": move_pct,
            "win": win,
        })

    return trades


def backtest_market():
    print(f"Fetching active {MARKET_TYPE.upper()} symbols from Binance...")
    symbols = get_tradable_symbols()
    if BACKTEST_SYMBOL_LIMIT:
        symbols = symbols[:BACKTEST_SYMBOL_LIMIT]

    print(
        f"Backtesting {len(symbols)} symbols on {TIMEFRAME} | "
        f"{BACKTEST_CANDLES} candles each | "
        f"{'STRICT MODE (score 6/6)' if STRICT_MODE else 'normal mode'} | "
        f"win = price moves >= {BACKTEST_MIN_MOVE_PCT}% in predicted direction "
        f"within {BACKTEST_FOLLOWTHROUGH} candles\n"
    )

    all_trades = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(backtest_symbol, sym): sym for sym in symbols}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            trades = future.result()
            all_trades.extend(trades)
            with _print_lock:
                print(f"[{i}/{len(symbols)}] {futures[future]}: {len(trades)} historical signals found")

    if not all_trades:
        print("\nKoi historical signal nahi mila in filters ke sath — filters shayad zyada tight hain.")
        return

    total = len(all_trades)
    wins = sum(1 for t in all_trades if t["win"])
    win_rate = wins / total * 100

    bull_trades = [t for t in all_trades if t["direction"] == "bullish"]
    bear_trades = [t for t in all_trades if t["direction"] == "bearish"]
    bull_win_rate = (sum(1 for t in bull_trades if t["win"]) / len(bull_trades) * 100) if bull_trades else 0
    bear_win_rate = (sum(1 for t in bear_trades if t["win"]) / len(bear_trades) * 100) if bear_trades else 0

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Total historical signals: {total}")
    print(f"Overall win rate:  {win_rate:.1f}%  ({wins}/{total})")
    print(f"Bullish win rate:  {bull_win_rate:.1f}%  ({len(bull_trades)} signals)")
    print(f"Bearish win rate:  {bear_win_rate:.1f}%  ({len(bear_trades)} signals)")
    print("\nNote: this is REAL historical performance of this exact filter set on")
    print("this exact timeframe. It is not a promise about future performance —")
    print("markets change, and a strategy that worked on the last N candles can")
    print("stop working. Use this number to compare settings, not as a guarantee.")


if __name__ == "__main__":
    if BACKTEST_MODE:
        backtest_market()
    else:
        scan_market()
