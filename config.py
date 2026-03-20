# ============================================================
# config.py — Edit this file to change what gets analyzed
# ============================================================

# ── Single-ticker analysis (used by Final.py) ────────────────
TICKER     = "AAPL"       # Stock symbol: "TSLA", "GOOGL", "AMZN", etc.
START_DATE = "2023-01-01" # Start of date range (YYYY-MM-DD)
END_DATE   = "2026-01-01" # End of date range   (YYYY-MM-DD)

# How many future trading days to predict
FORECAST_DAYS = 5

# ── Magnificent 7 (used by mag7_model.py and compare.py) ─────
MAG7       = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
MAG7_START = "2023-01-01"  # start earlier to offset ~20-day indicator warmup
MAG7_END   = "2026-01-01"

# ── Shared indicator settings (used by all scripts) ──────────
EMA_PERIODS = [9, 21, 50, 200]
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9