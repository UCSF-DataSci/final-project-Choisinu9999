# ============================================================
# Final.py — Stock Analysis Engine
# Edit config.py to change ticker/settings, not this file.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import config

plt.rcParams.update({
    "axes.facecolor":    "#111111",
    "figure.facecolor":  "#0d0d0d",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#cccccc",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "grid.color":        "#222222",
    "grid.linestyle":    "-",
    "text.color":        "white",
    "legend.facecolor":  "#1a1a1a",
    "legend.edgecolor":  "#333333",
    "legend.fontsize":   8,
    "axes.titlesize":    11,
    "axes.titlepad":     8,
    "axes.titlecolor":   "white",
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})

COLORS = {
    "price":    "#29B6F6",
    "ema9":     "#FFD54F",
    "ema21":    "#F06292",
    "ema50":    "#66BB6A",
    "ema200":   "#FF7043",
    "pvt":      "#CE93D8",
    "volume":   "#1E3A5F",
    "macd":     "#26C6DA",
    "signal":   "#FFA726",
    "hist_pos": "#43A047",
    "hist_neg": "#E53935",
    "rsi":      "#FFD54F",
    "lr":       "#AB47BC",
    "gb":       "#FF7043",
    "forecast": "#FF7043",
}


# ============================================================
# STEP 1: FETCH
# ============================================================

def fetch_data():
    ticker, start, end = config.TICKER, config.START_DATE, config.END_DATE
    print(f"\n📥 Fetching {ticker}  {start} → {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    print(f"   ✅ {len(df)} trading days loaded.\n")
    if len(df) < 60:
        print("⚠️  Warning: fewer than 60 days of data. Extend START_DATE in config.py for better results.")
    return df


# ============================================================
# STEP 2: INDICATORS
# ============================================================

def add_indicators(df):
    for p in config.EMA_PERIODS:
        df[f"EMA{p}"] = df["Close"].ewm(span=p, adjust=False).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(config.RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(config.RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    ef = df["Close"].ewm(span=config.MACD_FAST,   adjust=False).mean()
    es = df["Close"].ewm(span=config.MACD_SLOW,   adjust=False).mean()
    df["MACD"]        = ef - es
    df["MACD_Signal"] = df["MACD"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    df["PVT"] = (df["Close"].pct_change().fillna(0) * df["Volume"]).cumsum()

    df.dropna(inplace=True)
    return df


def resample_ohlc(df, freq):
    return df.resample(freq).agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna()


# ============================================================
# STEP 3: ML
# ============================================================

def build_features(df):
    """
    Build normalized, outlier-cleaned features suitable for
    fitting multiple tickers into the same model.

    Key design decisions:
    - Use PRICE-RELATIVE features only (%, ratios) so TSLA at $200
      and GOOGL at $180 are comparable — raw EMAs would just encode
      the ticker's price level, not the pattern.
    - IQR clipping removes extreme outliers (earnings gaps, flash crashes)
      without dropping rows — clips to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    - Target is also expressed as next-day % return so the model learns
      directional moves, not absolute levels.
    """
    df = df.copy()

    # ── Price-relative features (cross-ticker comparable) ────
    close = df["Close"]

    # EMA distance: how far price is from each EMA (%)
    for p in config.EMA_PERIODS:
        df[f"EMA{p}_dist"] = (close - df[f"EMA{p}"]) / df[f"EMA{p}"]

    daily_ret = close.pct_change()

    # Short, medium, long momentum (%)
    df["return_1"]  = daily_ret
    df["return_3"]  = close.pct_change(3)
    df["return_5"]  = close.pct_change(5)
    df["return_10"] = close.pct_change(10)
    df["return_20"] = close.pct_change(20)

    # Mean reversion signal: z-score of price vs 20-day mean
    roll_mean = close.rolling(20).mean()
    roll_std  = close.rolling(20).std()
    df["zscore_20"] = (close - roll_mean) / roll_std

    # Volatility at multiple windows
    df["vol_5"]  = daily_ret.rolling(5).std()
    df["vol_10"] = daily_ret.rolling(10).std()
    df["vol_20"] = daily_ret.rolling(20).std()

    # Volatility regime: is current vol higher than usual? (>1 = elevated)
    df["vol_regime"] = df["vol_5"] / df["vol_20"]

    # RSI distance from neutral
    df["rsi_dist"] = df["RSI"] - 50

    # RSI momentum: is RSI rising or falling?
    df["rsi_change"] = df["RSI"].diff(3)

    # MACD normalized by price
    df["macd_norm"]  = df["MACD"] / close
    df["macd_cross"] = (df["MACD"] - df["MACD_Signal"]) / close

    # MACD histogram direction (rising/falling)
    df["macd_hist_chg"] = df["MACD_Hist"].diff(1)

    # Volume ratio and volume momentum
    df["vol_ratio"]  = df["Volume"] / df["Volume"].rolling(20).mean()
    df["vol_chg"]    = df["Volume"].pct_change(5)

    # High-Low range normalized (daily candle body size)
    df["hl_range"]   = (df["High"] - df["Low"]) / close

    df.dropna(inplace=True)

    # ── IQR outlier clipping ──────────────────────────────────
    clip_cols = ["return_1","return_3","return_5","return_10","return_20",
                 "vol_5","vol_10","vol_20","vol_regime","zscore_20",
                 "rsi_change","macd_norm","macd_cross","macd_hist_chg",
                 "vol_ratio","vol_chg","hl_range"]
    for col in clip_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)

    features = ([f"EMA{p}_dist" for p in config.EMA_PERIODS] +
                ["RSI", "rsi_dist", "rsi_change",
                 "macd_norm", "macd_cross", "macd_hist_chg",
                 "return_1", "return_3", "return_5", "return_10", "return_20",
                 "vol_5", "vol_10", "vol_20", "vol_regime",
                 "zscore_20", "vol_ratio", "vol_chg", "hl_range"])
    return df, features


def train_models(df):
    df, features = build_features(df)

    # Target: next-day % return (price-relative → cross-ticker comparable)
    df["target"] = df["Close"].pct_change(1).shift(-1)
    df.dropna(inplace=True)

    X        = df[features].copy()
    y        = df["target"]
    close_ref = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    _, close_test = train_test_split(close_ref, test_size=0.2, shuffle=False)

    # StandardScaler: fit on train only, apply to test
    # This is the scaler you'd reuse across tickers — fit once on combined data
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    lr = LinearRegression().fit(Xtr, y_train)
    lr_ret = lr.predict(Xte)

    gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                                   min_samples_leaf=8, max_features=0.8,
                                   subsample=0.85, random_state=42)
    gb.fit(Xtr, y_train)
    gb_ret = gb.predict(Xte)

    # Metrics on RETURNS (honest evaluation)
    print("=" * 52)
    print(f"  Model Results — {config.TICKER}  (on out-of-sample returns)")
    print("=" * 52)
    print(f"  Linear Regression  MAE: {mean_absolute_error(y_test, lr_ret)*100:.4f}%   R²: {r2_score(y_test, lr_ret):.4f}")
    print(f"  Gradient Boosting  MAE: {mean_absolute_error(y_test, gb_ret)*100:.4f}%   R²: {r2_score(y_test, gb_ret):.4f}")
    print(f"  Features: {len(features)} price-relative, IQR-clipped")
    print(f"  Note: R² near 0 is normal for daily returns (efficient market)")
    print("=" * 52)

    # Convert returns → price for plotting only
    prev    = close_test.values
    lr_pred = prev * (1 + lr_ret)
    gb_pred = prev * (1 + gb_ret)
    y_price = prev * (1 + y_test.values)

    return dict(X_test=X_test,
                y_test=pd.Series(y_price, index=X_test.index),
                lr_pred=lr_pred, gb_pred=gb_pred,
                scaler=scaler, gb_model=gb, features=features)


def forecast_future(df, res):
    # Seed with recent prices so we can roll indicators forward
    recent_close = list(df["Close"].values[-60:])
    last_close   = recent_close[-1]
    last_volume  = float(df["Volume"].iloc[-1])

    preds = []
    dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1),
                           periods=config.FORECAST_DAYS)

    # Rolling state for EMAs
    ema_vals    = {p: float(df[f"EMA{p}"].iloc[-1]) for p in config.EMA_PERIODS}
    macd_fast   = float(df["Close"].ewm(span=config.MACD_FAST,   adjust=False).mean().iloc[-1])
    macd_slow   = float(df["Close"].ewm(span=config.MACD_SLOW,   adjust=False).mean().iloc[-1])
    macd_signal = float(df["MACD_Signal"].iloc[-1])
    prev_close  = last_close

    # RSI rolling state
    deltas   = df["Close"].diff().iloc[-config.RSI_PERIOD:]
    avg_gain = float(deltas.clip(lower=0).mean())
    avg_loss = float((-deltas.clip(upper=0)).mean())

    # Seed lag returns and volatility
    daily_rets = df["Close"].pct_change().dropna()
    ret1  = float(daily_rets.iloc[-1])
    ret3  = float(df["Close"].pct_change(3).iloc[-1])
    ret5  = float(df["Close"].pct_change(5).iloc[-1])
    ret10 = float(df["Close"].pct_change(10).iloc[-1])
    ret20 = float(df["Close"].pct_change(20).iloc[-1])
    vol5  = float(daily_rets.rolling(5).std().iloc[-1])
    vol10 = float(daily_rets.rolling(10).std().iloc[-1])
    vol20 = float(daily_rets.rolling(20).std().iloc[-1])
    prev_macd_hist = float(df["MACD_Hist"].iloc[-2])
    rsi_prev3 = float(df["RSI"].iloc[-4])
    last_high = float(df["High"].iloc[-1])
    last_low  = float(df["Low"].iloc[-1])

    for i in range(config.FORECAST_DAYS):
        macd_line      = macd_fast - macd_slow
        rs             = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi            = 100 - (100 / (1 + rs))
        rsi_d          = rsi - 50
        rsi_chg        = rsi - rsi_prev3
        macd_norm      = macd_line / prev_close if prev_close else 0
        macd_cross     = (macd_line - macd_signal) / prev_close if prev_close else 0
        macd_hist      = macd_line - macd_signal
        macd_hist_chg  = macd_hist - prev_macd_hist
        ema_dists      = [(prev_close - ema_vals[p]) / ema_vals[p] for p in config.EMA_PERIODS]
        vol_regime     = vol5 / vol20 if vol20 > 0 else 1.0
        vol_ratio      = 1.0
        vol_chg        = 0.0
        zscore_20      = (prev_close - float(df["Close"].rolling(20).mean().iloc[-1])) /                           max(float(df["Close"].rolling(20).std().iloc[-1]), 1e-8)
        hl_range       = (last_high - last_low) / prev_close if prev_close else 0

        features = (ema_dists +
                    [rsi, rsi_d, rsi_chg,
                     macd_norm, macd_cross, macd_hist_chg,
                     ret1, ret3, ret5, ret10, ret20,
                     vol5, vol10, vol20, vol_regime,
                     zscore_20, vol_ratio, vol_chg, hl_range])
        row = np.array(features).reshape(1, -1)

        # Predict next-day return, convert to price
        ret_pred = res["gb_model"].predict(res["scaler"].transform(row))[0]
        vol      = float(df["Close"].pct_change().std())
        ret_pred += np.random.normal(0, vol * 0.4)
        p = prev_close * (1 + ret_pred)

        # Anchor day 1 to last actual close
        blend = max(0, 1 - i / 3)
        p     = p + blend * (last_close - p)

        preds.append(p)

        # Roll all indicators forward
        alpha_fast   = 2 / (config.MACD_FAST   + 1)
        alpha_slow   = 2 / (config.MACD_SLOW   + 1)
        alpha_signal = 2 / (config.MACD_SIGNAL + 1)
        macd_fast    = alpha_fast   * p + (1 - alpha_fast)   * macd_fast
        macd_slow    = alpha_slow   * p + (1 - alpha_slow)   * macd_slow
        macd_signal  = alpha_signal * (macd_fast - macd_slow) + (1 - alpha_signal) * macd_signal
        for period in config.EMA_PERIODS:
            alpha = 2 / (period + 1)
            ema_vals[period] = alpha * p + (1 - alpha) * ema_vals[period]
        d        = p - prev_close
        avg_gain = (avg_gain * (config.RSI_PERIOD - 1) + max(d, 0))  / config.RSI_PERIOD
        avg_loss = (avg_loss * (config.RSI_PERIOD - 1) + max(-d, 0)) / config.RSI_PERIOD
        new_ret        = (p - prev_close) / prev_close if prev_close != 0 else 0
        ret20          = ret10
        ret10          = ret5
        ret5           = ret3
        ret3           = ret1
        ret1           = new_ret
        vol5           = vol5  * 0.8 + abs(new_ret) * 0.2
        vol10          = vol10 * 0.9 + abs(new_ret) * 0.1
        vol20          = vol20 * 0.95 + abs(new_ret) * 0.05
        prev_macd_hist = macd_hist
        rsi_prev3      = rsi
        last_high      = p * 1.005
        last_low       = p * 0.995
        prev_close     = p

    return pd.Series(preds, index=dates)


# ============================================================
# STEP 4: PLOT HELPERS
# ============================================================

def draw_candles(ax, data, color_up="#26A69A", color_down="#EF5350", bar_width=0.6):
    """
    Draw candlestick bars:
      - Thin wick  = High to Low range
      - Filled body = Open to Close
      - Green (hollow top) if Close >= Open, Red (filled) if Close < Open
    """
    dates_num = mdates.date2num(data.index.to_pydatetime())
    half_w    = bar_width / 2

    for dnum, row in zip(dates_num, data.itertuples()):
        color  = color_up if row.Close >= row.Open else color_down
        body_b = min(row.Open, row.Close)
        body_h = abs(row.Close - row.Open) or 0.01  # avoid zero height
        # Wick
        ax.plot([dnum, dnum], [row.Low, row.High], color=color, lw=0.8, zorder=2)
        # Body
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
            (dnum - half_w, body_b), bar_width, body_h,
            facecolor=color, edgecolor=color, lw=0.5, zorder=3
        ))


def fmt_axis(ax, df_index, price_series=None, rsi=False, mirror_y=True, force_daily=False):
    """Apply clean x-axis date format and fine y-axis increments."""
    # X-axis: short date labels
    span_days = (df_index[-1] - df_index[0]).days
    if force_daily or span_days <= 60:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # every Monday
    elif span_days > 500:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y %b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    elif span_days > 90:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Y-axis: fine increments based on price range
    if rsi:
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    elif price_series is not None:
        rng = float(price_series.max() - price_series.min())
        step = max(1.0, round(rng / 20, 1))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(step / 2))

    ax.grid(True, which="major", alpha=0.2)
    ax.grid(True, which="minor", alpha=0.07)

    # Mirror y-axis labels on right side (skip for twin axes)
    if mirror_y:
        ax.yaxis.set_ticks_position("both")
        ax.tick_params(axis="y", which="both", left=True, right=True,
                       labelleft=True, labelright=True)


# ============================================================
# STEP 5: PLOT
# ============================================================

def plot_all(df, res, forecast):
    ticker    = config.TICKER
    weekly    = resample_ohlc(df, "W")
    quarterly = resample_ohlc(df, "QE")

    from matplotlib.lines import Line2D

    # Taller figure + much more vertical breathing room between panels
    fig = plt.figure(figsize=(20, 56), facecolor="#0d0d0d")
    fig.suptitle(f"{ticker}  —  Full Technical Analysis",
                 fontsize=22, fontweight="bold", color="white", y=0.999)

    gs = gridspec.GridSpec(8, 1, figure=fig,
                           top=0.975, bottom=0.005,
                           hspace=1.1, left=0.08, right=0.97)

    ema_c = [COLORS["ema9"], COLORS["ema21"], COLORS["ema50"], COLORS["ema200"]]

    # ── Row 0: Daily Candlestick (last 120 days) + EMAs ──────
    # Limit to last 120 trading days so candles are actually visible
    df_daily = df.iloc[-120:].copy()
    ax = fig.add_subplot(gs[0])
    ax.xaxis_date()
    draw_candles(ax, df_daily, bar_width=0.5)
    for p, c in zip(config.EMA_PERIODS, ema_c):
        col = f"EMA{p}"
        if col in df_daily.columns:
            ax.plot(df_daily.index, df_daily[col], color=c, lw=1.0,
                    linestyle="--", label=f"EMA {p}", alpha=0.9)
    handles = [
        Line2D([0],[0], color="#26A69A", lw=2, label="Up candle"),
        Line2D([0],[0], color="#EF5350", lw=2, label="Down candle"),
    ] + [Line2D([0],[0], color=c, lw=1.2, linestyle="--", label=f"EMA {p}")
         for p, c in zip(config.EMA_PERIODS, ema_c)]
    ax.legend(handles=handles, ncol=6, loc="upper left", fontsize=9)
    ax.set_title("Daily Candlestick  +  EMAs  (last 120 trading days)", pad=10)
    ax.set_ylabel("Price (USD)")
    fmt_axis(ax, df_daily.index, df_daily["Close"])

    # ── Row 1: Weekly Candlestick ────────────────────────────
    ax = fig.add_subplot(gs[1])
    ax.xaxis_date()
    draw_candles(ax, weekly, bar_width=3.5)
    ax.legend(handles=[Line2D([0],[0],color="#26A69A",lw=2,label="Up candle"),
                        Line2D([0],[0],color="#EF5350",lw=2,label="Down candle")],
              loc="upper left", fontsize=9)
    ax.set_title("Weekly Candlestick", pad=10)
    ax.set_ylabel("Price (USD)")
    fmt_axis(ax, weekly.index, weekly["Close"])

    # ── Row 2: Quarterly Candlestick ────────────────────────
    ax = fig.add_subplot(gs[2])
    ax.xaxis_date()
    draw_candles(ax, quarterly, bar_width=22.0)
    ax.legend(handles=[Line2D([0],[0],color="#26A69A",lw=2,label="Up quarter"),
                        Line2D([0],[0],color="#EF5350",lw=2,label="Down quarter")],
              loc="upper left", fontsize=9)
    ax.set_title("Quarterly Candlestick  (Close = last trading day of quarter)", pad=10)
    ax.set_ylabel("Price (USD)")
    fmt_axis(ax, quarterly.index, quarterly["Close"])

    # ── Row 3: Volume + PVT ──────────────────────────────────
    ax  = fig.add_subplot(gs[3])
    ax2 = ax.twinx()
    ax.bar(df.index, df["Volume"], color=COLORS["volume"], width=1, alpha=0.7, label="Volume")
    ax2.plot(df.index, df["PVT"],  color=COLORS["pvt"],    lw=1.1, label="PVT")
    ax.set_title("Volume  &  PVT (Price Volume Trend)", pad=10)
    ax.set_ylabel("Volume", color=COLORS["volume"])
    ax2.set_ylabel("PVT",   color=COLORS["pvt"])
    ax2.tick_params(axis="y", labelcolor=COLORS["pvt"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc="upper left", fontsize=9)
    fmt_axis(ax, df.index, mirror_y=False)

    # ── Row 4: MACD ──────────────────────────────────────────
    ax = fig.add_subplot(gs[4])
    ax.plot(df.index, df["MACD"],        color=COLORS["macd"],   lw=1.3, label="MACD")
    ax.plot(df.index, df["MACD_Signal"], color=COLORS["signal"], lw=1.0,
            linestyle="--", label="Signal")
    hist_c = [COLORS["hist_pos"] if v >= 0 else COLORS["hist_neg"] for v in df["MACD_Hist"]]
    ax.bar(df.index, df["MACD_Hist"], color=hist_c, width=1, alpha=0.6, label="Histogram")
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_title("MACD  (12 / 26 / 9)", pad=10)
    ax.set_ylabel("MACD")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    fmt_axis(ax, df.index, df["MACD"])

    # ── Row 5: RSI ───────────────────────────────────────────
    ax = fig.add_subplot(gs[5])
    ax.plot(df.index, df["RSI"], color=COLORS["rsi"], lw=1.3,
            label=f"RSI ({config.RSI_PERIOD})")
    ax.axhline(70, color=COLORS["hist_neg"], lw=0.9, linestyle="--", label="Overbought 70")
    ax.axhline(30, color=COLORS["hist_pos"], lw=0.9, linestyle="--", label="Oversold 30")
    ax.axhline(50, color="#555555", lw=0.6, linestyle=":")
    ax.fill_between(df.index, df["RSI"], 70,
                    where=(df["RSI"] >= 70), alpha=0.15, color=COLORS["hist_neg"])
    ax.fill_between(df.index, df["RSI"], 30,
                    where=(df["RSI"] <= 30), alpha=0.15, color=COLORS["hist_pos"])
    ax.set_ylim(0, 100)
    ax.set_title(f"RSI  ({config.RSI_PERIOD})", pad=10)
    ax.set_ylabel("RSI")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    fmt_axis(ax, df.index, rsi=True)

    # ── Row 6: ML Prediction — zoom into test period only ────
    ax = fig.add_subplot(gs[6])
    test_idx  = res["X_test"].index
    y_test    = np.array(res["y_test"]).flatten()
    lr_pred   = np.array(res["lr_pred"]).flatten()
    gb_pred   = np.array(res["gb_pred"]).flatten()
    ax.plot(test_idx, y_test,  color="white",        lw=1.5, label="Actual price")
    ax.plot(test_idx, lr_pred, color=COLORS["lr"],   lw=1.2, linestyle="--", label="Linear Regression")
    ax.plot(test_idx, gb_pred, color=COLORS["gb"],   lw=1.4, label="Gradient Boosting")
    # Shade the error region
    ax.fill_between(test_idx, y_test, gb_pred, alpha=0.12, color=COLORS["gb"])
    ax.set_title("ML Prediction  —  Test Set  (last 20% of data)", pad=10)
    ax.set_ylabel("Price (USD)")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    all_vals = pd.Series(np.concatenate([y_test, lr_pred, gb_pred]))
    fmt_axis(ax, test_idx, all_vals)

    # ── Row 7: Future Forecast ───────────────────────────────
    ax = fig.add_subplot(gs[7])
    recent          = df["Close"].iloc[-90:]
    combined_prices = pd.concat([recent, forecast])
    ax.plot(recent.index,   recent,   color=COLORS["price"],    lw=1.5, label="Recent Close")
    ax.plot(forecast.index, forecast, color=COLORS["forecast"], lw=1.8,
            linestyle="--", label=f"{config.FORECAST_DAYS}-Day Forecast")
    ax.fill_between(forecast.index, forecast * 0.97, forecast * 1.03,
                    alpha=0.18, color=COLORS["forecast"], label="±3% band")
    ax.axvline(x=df.index[-1], color="#888888", lw=1.2, linestyle=":")
    ax.set_title(f"Future {config.FORECAST_DAYS}-Day Price Forecast", pad=10)
    ax.set_ylabel("Price (USD)")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    ax.set_xlim(combined_prices.index[0], combined_prices.index[-1])
    fmt_axis(ax, combined_prices.index, combined_prices, force_daily=True)

    out = f"{ticker}_analysis.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"\n📊 Saved: {out}")
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Determine which tickers to run
    # If config.MAG7 is set and has entries, loop through all of them.
    # Otherwise fall back to the single config.TICKER.
    tickers_to_run = config.MAG7 if hasattr(config, "MAG7") and config.MAG7 else [config.TICKER]
    start          = config.MAG7_START if hasattr(config, "MAG7_START") and config.MAG7 else config.START_DATE
    end            = config.MAG7_END   if hasattr(config, "MAG7_END")   and config.MAG7 else config.END_DATE

    print(f"\n🚀 Running analysis on: {tickers_to_run}")
    print(f"   Date range: {start} → {end}\n")

    for ticker in tickers_to_run:
        print("=" * 52)
        print(f"  Processing: {ticker}")
        print("=" * 52)

        # Temporarily override config values so all functions pick up
        # the current ticker and date range
        config.TICKER     = ticker
        config.START_DATE = start
        config.END_DATE   = end

        try:
            df       = fetch_data()
            df       = add_indicators(df)
            results  = train_models(df)
            forecast = forecast_future(df, results)
            plot_all(df, results, forecast)
            print(f"\n✅ {ticker} done!\n")
        except Exception as e:
            print(f"\n⚠️  Skipping {ticker} — error: {e}\n")
            continue

    print("\n🏁 All tickers complete!")