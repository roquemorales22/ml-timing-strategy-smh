# ML Timing Strategy — SMH 📈

A machine learning project exploring whether technical and volatility-based features can predict short-term price direction for **SMH (semiconductor ETF)** — and whether that signal, even if modest, can be turned into a validated trading edge through careful backtesting.

---

## 🔍 What It Does

- Feature engineering across momentum, volatility, and volume indicators (RSI, MACD, Bollinger Bands, GARCH-based conditional volatility)
- Walk-forward GARCH estimation to avoid look-ahead bias in volatility features
- VIF-driven multicollinearity analysis, computed strictly on training data to prevent validation/test leakage
- Binary classification (Gradient Boosting, XGBoost, CatBoost) to predict 5-day-forward price direction
- Validation-driven model selection — the test set is touched exactly once, for a single final evaluation
- Confidence-tier analysis testing whether predicted probability correlates with real-world accuracy
- Backtest comparing a naive fixed-schedule investor against a model-guided strategy, with all thresholds selected on validation data before the final out-of-sample test

---

## 🗂 Project Structure

- `main_notebook.ipynb` — Full pipeline: data collection, feature engineering, model training/selection, and backtesting
- `data/` — Cached price data snapshot (if frozen for reproducibility)

---

## ⚙️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · CatBoost · Arch (GARCH) · Matplotlib

---

## ▶️ How to Run

\```bash
pip install pandas numpy scikit-learn xgboost catboost arch matplotlib yfinance
\```

1. Open `main_notebook.ipynb` in Jupyter Notebook or VS Code.
2. Run all cells **in order, top to bottom** (Kernel → Restart & Run All) — several steps (GARCH volatility estimation, VIF-based feature selection, model comparison) depend on earlier cells running in sequence, and partial re-runs can produce inconsistent results.
3. Review the confidence-tier check and backtest results near the end of the notebook before drawing conclusions from the final metrics.
