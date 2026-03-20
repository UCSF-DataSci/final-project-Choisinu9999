# ============================================================
# Integrate_model.py — Integrated Mag 7 Linear Regression
# All settings pulled from config.py — edit that file only.
#
# Trains ONE Ridge model on all Mag 7 stocks combined, then
# evaluates each ticker individually (in & out of sample).
# Run: python3 Integrate_model.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config  # ← single source of truth

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

COLORS = {
    "AAPL":"#29B6F6", "MSFT":"#66BB6A", "GOOGL":"#FFD54F",
    "AMZN":"#FFA726", "NVDA":"#26C6DA", "META":"#AB47BC", "TSLA":"#EF5350",
}


# ============================================================
# STEP 1: FETCH
# ============================================================

def fetch_all():
    all_dfs = {}
    for ticker in config.MAG7:
        print(f"  📥 {ticker}...", end=" ", flush=True)
        df = yf.download(ticker, start=config.MAG7_START,
                         end=config.MAG7_END, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        if len(df) > 100:
            all_dfs[ticker] = df
            print(f"✅ {len(df)} days")
        else:
            print("⚠️  skipped — insufficient data")
    return all_dfs


# ============================================================
# STEP 2: INDICATORS
# ============================================================

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


# ============================================================
# STEP 3: FEATURES
# ============================================================

def build_features(df):
    """Price-relative, IQR-clipped features — cross-ticker comparable."""
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

    # IQR clip
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
# STEP 4: TRAIN COMBINED MODEL
# ============================================================

def train_combined(all_dfs):
    print(f"\n🔧 Building combined dataset from {len(all_dfs)} tickers...")
    X_all, y_all  = [], []
    ticker_data   = {}

    for ticker, df in all_dfs.items():
        X, y, close, features = build_features(df)
        X_all.append(X)
        y_all.append(y)
        ticker_data[ticker] = (X, y, close)

    X_combined = pd.concat(X_all)
    y_combined = pd.concat(y_all)
    print(f"   Total rows: {len(X_combined)}  |  Features: {len(features)}")

    # Time-ordered 80/20 split
    split   = int(len(X_combined) * 0.8)
    X_train = X_combined.iloc[:split]
    y_train = y_combined.iloc[:split]
    X_test  = X_combined.iloc[split:]
    y_test  = y_combined.iloc[split:]

    scaler  = StandardScaler()
    Xtr     = scaler.fit_transform(X_train)
    Xte     = scaler.transform(X_test)

    # Linear Regression
    lr       = LinearRegression().fit(Xtr, y_train)
    lr_pred  = lr.predict(Xte)

    # Ridge Regression (better for correlated features)
    ridge    = Ridge(alpha=1.0).fit(Xtr, y_train)
    ridge_pred = ridge.predict(Xte)

    print("\n" + "=" * 60)
    print(f"  Combined Model Results  ({config.MAG7_START} → {config.MAG7_END})")
    print("=" * 60)
    print(f"  Linear Regression  MAE: {mean_absolute_error(y_test, lr_pred)*100:.4f}%"
          f"   R²: {r2_score(y_test, lr_pred):.4f}")
    print(f"  Ridge Regression   MAE: {mean_absolute_error(y_test, ridge_pred)*100:.4f}%"
          f"   R²: {r2_score(y_test, ridge_pred):.4f}")
    print("=" * 60)

    # Per-ticker breakdown
    print(f"\n  Per-Ticker Performance (Ridge — full dataset):")
    print(f"  {'Ticker':<8} {'MAE':>10} {'R²':>10}  {'In-Sample MAE':>16} {'In-Sample R²':>14}")
    print("  " + "-" * 62)

    ticker_results = {}
    for ticker, (X, y, close) in ticker_data.items():
        Xs        = scaler.transform(X)
        pred      = ridge.predict(Xs)

        # Out-of-sample (last 20%)
        sp        = int(len(X) * 0.8)
        oos_mae   = mean_absolute_error(y.iloc[sp:],  pred[sp:])  * 100
        oos_r2    = r2_score(y.iloc[sp:],  pred[sp:])

        # In-sample (first 80%)
        is_mae    = mean_absolute_error(y.iloc[:sp],  pred[:sp])  * 100
        is_r2     = r2_score(y.iloc[:sp],  pred[:sp])

        # Price conversion for plotting
        prev         = close.values[:-1]
        pred_price   = prev * (1 + pred[:-1])
        actual_price = prev * (1 + y.values[:-1])

        ticker_results[ticker] = {
            "oos_mae": oos_mae, "oos_r2": oos_r2,
            "is_mae":  is_mae,  "is_r2":  is_r2,
            "pred":    pred_price,
            "actual":  actual_price,
            "dates":   X.index[:-1],
        }
        print(f"  {ticker:<8} {oos_mae:>9.4f}%  {oos_r2:>10.4f}  "
              f"{is_mae:>14.4f}%  {is_r2:>13.4f}")
    print("  " + "-" * 62)

    return lr, ridge, scaler, features, ticker_results


# ============================================================
# STEP 5: PLOT PREDICTIONS
# ============================================================

def plot_predictions(ticker_results):
    n   = len(ticker_results)
    fig = plt.figure(figsize=(18, 4.5 * n), facecolor="#0d0d0d")
    fig.suptitle(
        f"Mag 7 — Integrated Ridge Model  ({config.MAG7_START} → {config.MAG7_END})\n"
        f"Tickers: {', '.join(config.MAG7)}",
        fontsize=16, fontweight="bold", color="white", y=1.005
    )
    gs = gridspec.GridSpec(n, 1, figure=fig,
                           top=0.975, bottom=0.02,
                           hspace=0.85, left=0.07, right=0.97)

    for i, (ticker, res) in enumerate(ticker_results.items()):
        ax = fig.add_subplot(gs[i])
        ax.plot(res["dates"], res["actual"], color="white", lw=1.2,
                label="Actual", alpha=0.9)
        ax.plot(res["dates"], res["pred"],   color=COLORS[ticker], lw=1.0,
                linestyle="--", label="Predicted (Ridge)", alpha=0.85)
        ax.fill_between(res["dates"], res["actual"], res["pred"],
                        alpha=0.1, color=COLORS[ticker])
        ax.set_title(
            f"{ticker}   OOS MAE: {res['oos_mae']:.4f}%   OOS R²: {res['oos_r2']:.4f}"
            f"   |   In-Sample MAE: {res['is_mae']:.4f}%   In-Sample R²: {res['is_r2']:.4f}",
            color=COLORS[ticker], fontsize=10, pad=6
        )
        ax.set_ylabel("Price (USD)", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.15)
        ax.tick_params(colors="#888888")

    out = "Integrate_model_predictions.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"\n📊 Saved: {out}")
    plt.close("all")


# ============================================================
# STEP 6: FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(ridge, features):
    coefs   = np.abs(ridge.coef_)
    idx     = np.argsort(coefs)[::-1][:15]
    labels  = [features[i] for i in idx][::-1]
    values  = coefs[idx][::-1]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#111111")
    ax.barh(labels, values, color="#29B6F6", alpha=0.85)
    ax.set_title("Top 15 Feature Importances — Ridge |Coefficients|",
                 color="white", fontsize=13)
    ax.set_xlabel("|Coefficient|", color="#cccccc")
    ax.tick_params(colors="#888888")
    ax.grid(True, axis="x", alpha=0.15)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    plt.tight_layout()

    out = "Integrate_model_features.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"📊 Saved: {out}")
    plt.close("all")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Integrate_model.py — Mag 7 Combined Ridge Regression")
    print(f"  Tickers  : {config.MAG7}")
    print(f"  Date range: {config.MAG7_START} → {config.MAG7_END}")
    print(f"  EMA periods: {config.EMA_PERIODS}   RSI: {config.RSI_PERIOD}")
    print("=" * 60)

    print("\n📥 Fetching data...")
    all_dfs = fetch_all()

    for ticker in list(all_dfs.keys()):
        all_dfs[ticker] = add_indicators(all_dfs[ticker])

    lr, ridge, scaler, features, ticker_results = train_combined(all_dfs)
    plot_predictions(ticker_results)
    plot_feature_importance(ridge, features)

    print("\n✅ Done!")
    print("   Outputs: Integrate_model_predictions.png, Integrate_model_features.png")