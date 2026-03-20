# ============================================================
# compare.py — Strategy Comparison
#
# Compares two prediction strategies on each Mag 7 stock:
#   Strategy A: Single-ticker model (trained on that stock only)
#   Strategy B: Combined Mag 7 model (trained on all 7 stocks)
#
# Run: python3 compare.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import config

# ── Colors ───────────────────────────────────────────────────
TICKER_COLORS = {
    "AAPL":"#29B6F6", "MSFT":"#66BB6A", "GOOGL":"#FFD54F",
    "AMZN":"#FFA726", "NVDA":"#26C6DA", "META":"#AB47BC", "TSLA":"#EF5350",
}
COL_A = "#FF7043"   # Strategy A color
COL_B = "#26C6DA"   # Strategy B color

plt.rcParams.update({
    "axes.facecolor":   "#111111",
    "figure.facecolor": "#0d0d0d",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "grid.color":       "#222222",
    "text.color":       "white",
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#333333",
    "legend.fontsize":  8,
})


# ============================================================
# SHARED UTILITIES
# ============================================================

def fetch(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    return df


def add_indicators(df):
    close = df["Close"]
    for p in config.EMA_PERIODS:
        df[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(config.RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(config.RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ef = close.ewm(span=config.MACD_FAST,   adjust=False).mean()
    es = close.ewm(span=config.MACD_SLOW,   adjust=False).mean()
    df["MACD"]        = ef - es
    df["MACD_Signal"] = df["MACD"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    df.dropna(inplace=True)
    return df


def build_features(df):
    """Price-relative features — comparable across all tickers."""
    df    = df.copy()
    close = df["Close"]
    dr    = close.pct_change()

    for p in config.EMA_PERIODS:
        df[f"EMA{p}_dist"] = (close - df[f"EMA{p}"]) / df[f"EMA{p}"]

    df["return_1"]      = dr
    df["return_3"]      = close.pct_change(3)
    df["return_5"]      = close.pct_change(5)
    df["return_10"]     = close.pct_change(10)
    df["return_20"]     = close.pct_change(20)
    df["vol_5"]         = dr.rolling(5).std()
    df["vol_10"]        = dr.rolling(10).std()
    df["vol_20"]        = dr.rolling(20).std()
    df["vol_regime"]    = df["vol_5"] / df["vol_20"]
    df["zscore_20"]     = (close - close.rolling(20).mean()) / close.rolling(20).std()
    df["rsi_dist"]      = df["RSI"] - 50
    df["rsi_change"]    = df["RSI"].diff(3)
    df["macd_norm"]     = df["MACD"] / close
    df["macd_cross"]    = (df["MACD"] - df["MACD_Signal"]) / close
    df["macd_hist_chg"] = df["MACD_Hist"].diff(1)
    df["vol_ratio"]     = df["Volume"] / df["Volume"].rolling(20).mean()
    df["hl_range"]      = (df["High"] - df["Low"]) / close
    df.dropna(inplace=True)

    clip_cols = ["return_1","return_3","return_5","return_10","return_20",
                 "vol_5","vol_10","vol_20","vol_regime","zscore_20",
                 "rsi_change","macd_norm","macd_cross","macd_hist_chg",
                 "vol_ratio","hl_range"]
    for col in clip_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        df[col]= df[col].clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)

    features = ([f"EMA{p}_dist" for p in config.EMA_PERIODS] +
                ["RSI","rsi_dist","rsi_change",
                 "macd_norm","macd_cross","macd_hist_chg",
                 "return_1","return_3","return_5","return_10","return_20",
                 "vol_5","vol_10","vol_20","vol_regime",
                 "zscore_20","vol_ratio","hl_range"])

    df["target"] = close.pct_change(1).shift(-1)
    df.dropna(inplace=True)
    return df[features], df["target"], df["Close"], features


# ============================================================
# STRATEGY A — Single-ticker Ridge model
# ============================================================

def strategy_a(ticker, raw_df):
    """Train and evaluate a model on ONE ticker only."""
    X, y, close, features = build_features(raw_df)

    split   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    close_test      = close.iloc[split:]

    scaler = StandardScaler()
    Xtr    = scaler.fit_transform(X_train)
    Xte    = scaler.transform(X_test)

    model  = Ridge(alpha=1.0).fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mae = mean_absolute_error(y_test, y_pred) * 100
    r2  = r2_score(y_test, y_pred)

    prev         = close_test.values[:-1]
    pred_price   = prev * (1 + y_pred[:-1])
    actual_price = prev * (1 + y_test.values[:-1])
    dates        = X_test.index[:-1]

    return {"mae":mae, "r2":r2, "pred":pred_price,
            "actual":actual_price, "dates":dates, "label":"Strategy A: Single-Ticker"}


# ============================================================
# STRATEGY B — Combined Mag 7 Ridge model
# ============================================================

def strategy_b_train(all_raw):
    """Train one Ridge model on all 7 tickers combined."""
    X_all, y_all = [], []
    for ticker, df in all_raw.items():
        X, y, _, _ = build_features(df)
        X_all.append(X)
        y_all.append(y)

    X_combined = pd.concat(X_all)
    y_combined = pd.concat(y_all)

    # Time-ordered 80/20 split across the combined dataset
    split   = int(len(X_combined) * 0.8)
    X_train = X_combined.iloc[:split]
    y_train = y_combined.iloc[:split]

    scaler  = StandardScaler()
    Xtr     = scaler.fit_transform(X_train)

    model   = Ridge(alpha=1.0).fit(Xtr, y_train)
    print(f"   Combined model trained on {len(X_train)} rows from {len(all_raw)} tickers")
    return model, scaler


def strategy_b_eval(ticker, raw_df, model, scaler):
    """Evaluate the shared Mag 7 model on a single ticker."""
    X, y, close, _ = build_features(raw_df)

    # Use same 20% test split as Strategy A for fair comparison
    split    = int(len(X) * 0.8)
    X_test   = X.iloc[split:]
    y_test   = y.iloc[split:]
    close_test = close.iloc[split:]

    Xte    = scaler.transform(X_test)
    y_pred = model.predict(Xte)

    mae = mean_absolute_error(y_test, y_pred) * 100
    r2  = r2_score(y_test, y_pred)

    prev         = close_test.values[:-1]
    pred_price   = prev * (1 + y_pred[:-1])
    actual_price = prev * (1 + y_test.values[:-1])
    dates        = X_test.index[:-1]

    return {"mae":mae, "r2":r2, "pred":pred_price,
            "actual":actual_price, "dates":dates, "label":"Strategy B: Mag 7 Combined"}


# ============================================================
# PLOT — side-by-side comparison per ticker
# ============================================================

def plot_comparison(results):
    """
    For each ticker: left panel = Strategy A, right panel = Strategy B.
    Bottom summary bar: MAE and R² side by side.
    """
    tickers = list(results.keys())
    n       = len(tickers)

    fig = plt.figure(figsize=(18, 5 * n), facecolor="#0d0d0d")
    fig.suptitle("Strategy Comparison: Single-Ticker vs Mag 7 Combined Model",
                 fontsize=18, fontweight="bold", color="white", y=1.005)

    gs = gridspec.GridSpec(n, 2, figure=fig,
                           top=0.985, bottom=0.02,
                           hspace=0.75, wspace=0.25,
                           left=0.06, right=0.97)

    for i, ticker in enumerate(tickers):
        a = results[ticker]["A"]
        b = results[ticker]["B"]
        tc = TICKER_COLORS[ticker]

        for j, (res, col, label) in enumerate([(a, COL_A, "A"), (b, COL_B, "B")]):
            ax = fig.add_subplot(gs[i, j])
            ax.plot(res["dates"], res["actual"], color="white",   lw=1.2,
                    label="Actual", alpha=0.9)
            ax.plot(res["dates"], res["pred"],   color=col,       lw=1.0,
                    linestyle="--", label=f"Predicted", alpha=0.85)
            ax.fill_between(res["dates"], res["actual"], res["pred"],
                            alpha=0.1, color=col)

            better = "✅" if (label == "A" and a["mae"] <= b["mae"]) or \
                             (label == "B" and b["mae"] <= a["mae"]) else ""
            ax.set_title(
                f"{ticker}  {res['label']}  {better}\n"
                f"MAE: {res['mae']:.4f}%   R²: {res['r2']:.4f}",
                color=col, fontsize=10, pad=6
            )
            ax.set_ylabel("Price (USD)", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.15)

    plt.savefig("comparison.png", dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    print("\n📊 Saved: comparison.png")
    plt.show()


def plot_summary(results):
    """Bar chart: MAE and R² for each ticker, Strategy A vs B."""
    tickers = list(results.keys())
    x       = np.arange(len(tickers))
    w       = 0.35

    mae_a = [results[t]["A"]["mae"] for t in tickers]
    mae_b = [results[t]["B"]["mae"] for t in tickers]
    r2_a  = [results[t]["A"]["r2"]  for t in tickers]
    r2_b  = [results[t]["B"]["r2"]  for t in tickers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), facecolor="#0d0d0d")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#888888")
        ax.grid(True, axis="y", alpha=0.15)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    # MAE
    ax1.bar(x - w/2, mae_a, w, color=COL_A, alpha=0.85, label="Strategy A: Single-Ticker")
    ax1.bar(x + w/2, mae_b, w, color=COL_B, alpha=0.85, label="Strategy B: Mag 7 Combined")
    ax1.set_xticks(x); ax1.set_xticklabels(tickers)
    ax1.set_title("MAE (%) — lower is better", color="white", fontsize=13)
    ax1.set_ylabel("MAE (%)", color="#cccccc")
    ax1.legend()

    # R²
    ax2.bar(x - w/2, r2_a, w, color=COL_A, alpha=0.85, label="Strategy A: Single-Ticker")
    ax2.bar(x + w/2, r2_b, w, color=COL_B, alpha=0.85, label="Strategy B: Mag 7 Combined")
    ax2.set_xticks(x); ax2.set_xticklabels(tickers)
    ax2.set_title("R² — higher is better", color="white", fontsize=13)
    ax2.set_ylabel("R²", color="#cccccc")
    ax2.legend()

    fig.suptitle("Strategy A vs Strategy B — Summary", fontsize=15,
                 fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig("comparison_summary.png", dpi=130, bbox_inches="tight",
                facecolor="#0d0d0d")
    print("📊 Saved: comparison_summary.png")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 58)
    print("  Strategy Comparison — Single vs Mag 7 Combined")
    print("=" * 58)

    # Fetch all tickers
    print("\n📥 Fetching Mag 7 data...")
    all_raw = {}
    for ticker in config.MAG7:
        print(f"  {ticker}...", end=" ", flush=True)
        df = fetch(ticker, config.MAG7_START, config.MAG7_END)
        df = add_indicators(df)
        if len(df) > 100:
            all_raw[ticker] = df
            print(f"✅ {len(df)} days")
        else:
            print("⚠️  skipped")

    # Train combined Mag 7 model (Strategy B)
    print("\n🔧 Training Strategy B (combined Mag 7 model)...")
    b_model, b_scaler = strategy_b_train(all_raw)

    # Evaluate both strategies on each ticker
    print("\n📊 Evaluating both strategies...\n")
    print(f"  {'Ticker':<8} {'Strategy A MAE':>16} {'A R²':>8} {'Strategy B MAE':>16} {'B R²':>8} {'Winner':>8}")
    print("  " + "-" * 66)

    results = {}
    for ticker, df in all_raw.items():
        res_a = strategy_a(ticker, df)
        res_b = strategy_b_eval(ticker, df, b_model, b_scaler)
        results[ticker] = {"A": res_a, "B": res_b}
        winner = "A" if res_a["mae"] <= res_b["mae"] else "B"
        print(f"  {ticker:<8} {res_a['mae']:>14.4f}%  {res_a['r2']:>8.4f}  "
              f"{res_b['mae']:>14.4f}%  {res_b['r2']:>8.4f}  {'→ '+winner:>8}")

    print("  " + "-" * 66)

    # Count wins
    wins_a = sum(1 for t in results if results[t]["A"]["mae"] <= results[t]["B"]["mae"])
    wins_b = len(results) - wins_a
    print(f"\n  Strategy A wins: {wins_a}/{len(results)} tickers")
    print(f"  Strategy B wins: {wins_b}/{len(results)} tickers")

    if wins_a > wins_b:
        print("\n  ✅ Overall winner: Strategy A (Single-Ticker)")
        print("     Specialization beats generalization for these stocks.")
    elif wins_b > wins_a:
        print("\n  ✅ Overall winner: Strategy B (Mag 7 Combined)")
        print("     Cross-ticker patterns improve generalization.")
    else:
        print("\n  🤝 Tie — both strategies perform equally across the Mag 7.")

    plot_comparison(results)
    plot_summary(results)
    print("\n✅ Done!")