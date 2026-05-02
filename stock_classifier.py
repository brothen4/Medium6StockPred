"""
Stock Market Snapshot Classifier
----------------------------------
Predicts whether current market conditions are BULLISH or BEARISH
using a Random Forest classifier trained on technical indicators.

Usage:
If you want to keep inside downloads 
cd Downloads
    python stock_classifier.py                        # analyze default tickers
    python stock_classifier.py --ticker AAPL MSFT     # custom tickers
    python stock_classifier.py --train                # retrain model
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
from datetime import datetime, timedelta


# CONFIG

FORWARD_WINDOW   = 20       # days ahead we look to define label
BULLISH_THRESH   = 0.03     # >3% return in next FORWARD_WINDOW days → BULLISH
MODEL_PATH       = "rf_model.pkl"
SCALER_PATH      = "scaler.pkl"
DEFAULT_TICKERS  = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA"]

# FEATURE ENGINEERING
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features from OHLCV data.
    All features are derived from publicly available price/volume data.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    feat = pd.DataFrame(index=df.index)

    # ── Trend features ──────────────────────────────
    feat["sma_5"]       = close.rolling(5).mean()
    feat["sma_20"]      = close.rolling(20).mean()
    feat["sma_50"]      = close.rolling(50).mean()
    feat["ema_12"]      = close.ewm(span=12, adjust=False).mean()
    feat["ema_26"]      = close.ewm(span=26, adjust=False).mean()

    feat["price_vs_sma20"]  = (close - feat["sma_20"]) / feat["sma_20"]
    feat["price_vs_sma50"]  = (close - feat["sma_50"]) / feat["sma_50"]
    feat["sma5_vs_sma20"]   = (feat["sma_5"]  - feat["sma_20"]) / feat["sma_20"]
    feat["sma20_vs_sma50"]  = (feat["sma_20"] - feat["sma_50"]) / feat["sma_50"]

    # MACD
    feat["macd"]        = feat["ema_12"] - feat["ema_26"]
    feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
    feat["macd_hist"]   = feat["macd"] - feat["macd_signal"]

    # ── Momentum features ───────────────────────────
    feat["roc_5"]   = close.pct_change(5)
    feat["roc_10"]  = close.pct_change(10)
    feat["roc_20"]  = close.pct_change(20)

    # RSI (14-day)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.rolling(14).mean()
    avg_l = loss.rolling(14).mean()
    rs    = avg_g / (avg_l + 1e-9)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # ── Volatility features ─────────────────────────
    feat["atr_14"] = (
        pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        .rolling(14).mean()
    )
    feat["bb_width"] = (
        close.rolling(20).std() * 2 / (feat["sma_20"] + 1e-9)
    )
    feat["daily_return"]  = close.pct_change()
    feat["volatility_10"] = feat["daily_return"].rolling(10).std()

    # ── Volume features ─────────────────────────────
    vol_sma20 = volume.rolling(20).mean()
    feat["vol_ratio"]    = volume / (vol_sma20 + 1e-9)
    feat["vol_change"]   = volume.pct_change()

    # ── Candle-pattern proxies ───────────────────────
    feat["body_size"]   = (df["Close"] - df["Open"]).abs() / (df["Open"] + 1e-9)
    feat["upper_wick"]  = (high  - df[["Close","Open"]].max(axis=1)) / (df["Open"] + 1e-9)
    feat["lower_wick"]  = (df[["Close","Open"]].min(axis=1) - low)   / (df["Open"] + 1e-9)

    return feat


def build_labels(df: pd.DataFrame, forward_window: int = FORWARD_WINDOW,
                 threshold: float = BULLISH_THRESH) -> pd.Series:
    """
    Ground-truth label:
        BULLISH (1) if close price FORWARD_WINDOW days later is
                    > threshold % above today's close.
        BEARISH (0) otherwise.

    This is a look-forward label — only valid for training on historical data.
    """
    future_close  = df["Close"].shift(-forward_window)
    forward_return = (future_close - df["Close"]) / df["Close"]
    return (forward_return > threshold).astype(int)

# DATA COLLECTION
def fetch_data(tickers: list, period: str = "5y") -> pd.DataFrame:
    """Download historical OHLCV data via yfinance and build feature matrix."""
    all_frames = []
    print(f"\n📥  Fetching {period} of data for {len(tickers)} ticker(s)...")

    for ticker in tickers:
        try:
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if raw.empty or len(raw) < 100:
                print(f"   ⚠  {ticker}: insufficient data, skipping")
                continue

            # flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            features = compute_features(raw)
            labels   = build_labels(raw)

            combined = features.copy()
            combined["label"]  = labels
            combined["ticker"] = ticker
            combined["close"]  = raw["Close"]
            combined["date"]   = raw.index

            all_frames.append(combined)
            print(f"   ✓  {ticker}: {len(combined):,} rows")

        except Exception as e:
            print(f"   ✗  {ticker}: {e}")

    if not all_frames:
        raise RuntimeError("No data collected. Check your tickers / internet connection.")

    full = pd.concat(all_frames, ignore_index=True)
    full = full.dropna()
    full = full[full["label"].notna()]
    return full


# MODEL TRAINING

FEATURE_COLS = [
    "sma5_vs_sma20", "sma20_vs_sma50", "price_vs_sma20", "price_vs_sma50",
    "macd", "macd_signal", "macd_hist",
    "roc_5", "roc_10", "roc_20",
    "rsi_14",
    "atr_14", "bb_width", "volatility_10",
    "vol_ratio", "vol_change",
    "body_size", "upper_wick", "lower_wick",
]


def train_model(df: pd.DataFrame):
    """Train a Random Forest classifier and save model + scaler."""
    X = df[FEATURE_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  Model Accuracy : {acc:.1%}")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=["BEARISH","BULLISH"]))

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✓ Model saved → {MODEL_PATH}")

    return model, scaler, X_test, y_test, y_pred, df.loc[X_test.index]

# ANALYSIS HELPERS
def investment_horizon(proba_bullish: float) -> str:
    """Map confidence to a suggested investment horizon."""
    if proba_bullish >= 0.80:
        return "Long-term hold (3–12 months)"
    elif proba_bullish >= 0.65:
        return "Medium-term (1–3 months)"
    elif proba_bullish >= 0.55:
        return "Short-term swing (2–4 weeks)"
    else:
        return "Not recommended — wait for clearer signal"


def snapshot_prediction(ticker: str, model, scaler) -> dict:
    """Pull the most recent data for a ticker and return a live prediction."""
    raw = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
    if raw.empty or len(raw) < 60:
        return {"ticker": ticker, "error": "Insufficient recent data"}

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    feat  = compute_features(raw).dropna()
    if feat.empty:
        return {"ticker": ticker, "error": "Could not compute features"}

    latest = feat.iloc[[-1]][FEATURE_COLS]
    scaled = scaler.transform(latest)

    proba  = model.predict_proba(scaled)[0]
    pred   = model.predict(scaled)[0]
    label  = "🐂 BULLISH" if pred == 1 else "🐻 BEARISH"
    conf   = proba[pred]

    return {
        "ticker":    ticker,
        "date":      feat.index[-1].strftime("%Y-%m-%d"),
        "signal":    label,
        "confidence": conf,
        "proba_bull": proba[1],
        "horizon":   investment_horizon(proba[1]),
        "rsi":       round(float(feat["rsi_14"].iloc[-1]), 1),
        "macd":      round(float(feat["macd"].iloc[-1]), 4),
        "vol_ratio": round(float(feat["vol_ratio"].iloc[-1]), 2),
    }

# PLOTTING

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
                xticklabels=["BEARISH","BULLISH"],
                yticklabels=["BEARISH","BULLISH"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Market Classifier")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("  📊 Saved → confusion_matrix.png")


def plot_feature_importance(model):
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71" if v > importances.median() else "#e74c3c"
              for v in importances.values]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Top Feature Importances — Random Forest")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("  📊 Saved → feature_importance.png")


def plot_wrong_predictions(wrong_df: pd.DataFrame):
    """Show 5 misclassified samples with price context."""
    sample = wrong_df.head(5)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle("5 Misclassified Samples", fontsize=14, fontweight="bold")

    colors = {"🐂 BULLISH": "#2ecc71", "🐻 BEARISH": "#e74c3c"}

    for i, (idx, row) in enumerate(sample.iterrows()):
        ax = axes[i]
        true_lbl  = "🐂 BULLISH" if row["label"] == 1 else "🐻 BEARISH"
        pred_lbl  = "🐂 BULLISH" if row["predicted"] == 1 else "🐻 BEARISH"

        bar_data  = [row.get("rsi_14", 50), abs(row.get("macd", 0)) * 100,
                     row.get("vol_ratio", 1) * 20]
        bar_lbls  = ["RSI/2", "|MACD|×100", "Vol×20"]

        ax.bar(bar_lbls, bar_data,
               color=[colors.get(pred_lbl, "#95a5a6")] * 3, alpha=0.8)
        ax.set_title(
            f"#{i+1} {row.get('ticker','?')} {str(row.get('date',''))[:10]}\n"
            f"True: {true_lbl}\nPred: {pred_lbl}",
            fontsize=8
        )
        ax.set_ylim(0, 120)
        ax.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    plt.savefig("wrong_predictions.png", dpi=150)
    plt.close()
    print("  📊 Saved → wrong_predictions.png")


def plot_snapshot_results(results: list):
    """Bar chart of bullish probabilities for snapshot tickers."""
    df = pd.DataFrame([r for r in results if "error" not in r])
    if df.empty:
        return
    df = df.sort_values("proba_bull", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2ecc71" if v >= 0.5 else "#e74c3c" for v in df["proba_bull"]]
    bars = ax.barh(df["ticker"], df["proba_bull"], color=colors, alpha=0.85)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.2, label="50% threshold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of BULLISH signal")
    ax.set_title("Market Snapshot — Bullish Probability by Ticker")
    ax.legend()

    for bar, val in zip(bars, df["proba_bull"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.0%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("snapshot_results.png", dpi=150)
    plt.close()
    print("  📊 Saved → snapshot_results.png")

# WRONG PREDICTION ANALYSIS

def find_wrong_predictions(df_test: pd.DataFrame, y_test, y_pred) -> pd.DataFrame:
    """Return rows where the model was wrong, with context."""
    result = df_test.copy()
    result["label"]     = y_test.values
    result["predicted"] = y_pred
    wrong = result[result["label"] != result["predicted"]].copy()
    return wrong

# MAIN

def main():
    parser = argparse.ArgumentParser(description="Stock Market Bullish/Bearish Classifier")
    parser.add_argument("--tickers",  nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--train",    action="store_true", help="Force retrain model")
    parser.add_argument("--period",   default="5y", help="Training data period (e.g. 3y, 5y)")
    args = parser.parse_args()

    print("\n" + "═"*55)
    print("  📈  STOCK MARKET SNAPSHOT CLASSIFIER")
    print("═"*55)

    # ── Step 1: Train or load model ──────────────────
    if args.train or not os.path.exists(MODEL_PATH):
        print("\n🔧  TRAINING MODE")
        df = fetch_data(args.tickers, period=args.period)

        print(f"\n  Dataset size : {len(df):,} samples")
        print(f"  Bullish      : {df['label'].mean():.1%}")
        print(f"  Bearish      : {1-df['label'].mean():.1%}")

        model, scaler, X_test, y_test, y_pred, df_test = train_model(df)

        # Plots
        print("\n📊  Generating plots...")
        plot_confusion_matrix(y_test, y_pred)
        plot_feature_importance(model)

        wrong = find_wrong_predictions(df_test, y_test, y_pred)
        plot_wrong_predictions(wrong)

        print(f"\n  ⚠  Wrong predictions : {len(wrong):,} ({len(wrong)/len(y_test):.1%} of test set)")
        print("\n  Top 5 Misclassified Samples:")
        print("  " + "─"*70)
        for i, (_, row) in enumerate(wrong.head(5).iterrows()):
            true_lbl = "BULLISH" if row["label"]     == 1 else "BEARISH"
            pred_lbl = "BULLISH" if row["predicted"] == 1 else "BEARISH"
            print(f"  #{i+1} {str(row.get('ticker','?')):6s} "
                  f"{str(row.get('date',''))[:10]}  |  "
                  f"True: {true_lbl:7s}  Pred: {pred_lbl:7s}  "
                  f"RSI: {row.get('rsi_14',0):.1f}  "
                  f"MACD: {row.get('macd',0):.4f}")
    else:
        print(f"\n  ✓ Loading saved model from {MODEL_PATH}")
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

    # ── Step 2: Live snapshot predictions ────────────
    print("\n\n🔭  LIVE MARKET SNAPSHOT")
    print("  " + "─"*60)
    results = []
    for ticker in args.tickers:
        r = snapshot_prediction(ticker, model, scaler)
        results.append(r)
        if "error" in r:
            print(f"  {ticker:6s} ⚠  {r['error']}")
        else:
            bull_bar = "█" * int(r["proba_bull"] * 20)
            bear_bar = "░" * (20 - int(r["proba_bull"] * 20))
            print(f"\n  {r['ticker']:6s}  {r['signal']}  ({r['confidence']:.0%} confident)")
            print(f"  {'':6s}  [{bull_bar}{bear_bar}] {r['proba_bull']:.0%} bullish")
            print(f"  {'':6s}  RSI: {r['rsi']}  |  MACD: {r['macd']}  |  Vol Ratio: {r['vol_ratio']}")
            print(f"  {'':6s}  💡 Recommendation: {r['horizon']}")

    plot_snapshot_results(results)

    print("\n" + "═"*55)
    print("  ✅  Done! Charts saved to current directory.")
    print("═"*55 + "\n")


if __name__ == "__main__":
    main()
