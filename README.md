# ML Timing Strategy — SMH 📈

A machine learning project exploring whether technical and volatility-based features can predict short-term price direction for **SMH (semiconductor ETF)** — and whether that signal, even if modest, can be turned into a validated trading edge through careful backtesting.

---

## 🔍 What It Does

- Feature engineering across momentum, volatility, and volume indicators (RSI, MACD, Bollinger Bands, GARCH-based conditional volatility)
- Walk-forward GARCH estimation to avoid look-ahead bias in volatility features
- VIF-driven multicollinearity analysis, computed strictly on training data to prevent validation/test leakage
- Binary classification (Gradient Boosting, XGBoost, CatBoost) to predict 5-day-forward price direction
- Validation-driven model selection — the test set is touched exactly once, for a single final evaluation
- Multi-seed ablation of `garch_vol` on the selected model, to check whether its importance exceeds seed-to-seed noise
- Confidence-tier analysis testing whether predicted probability correlates with real-world accuracy
- Backtest comparing a naive fixed-schedule investor against a model-guided strategy, with all thresholds selected on validation data before the final out-of-sample test
- An exploratory confidence-tiered position-sizing backtest, plus a random-timing permutation test (500 within-year shuffles of the model's probabilities) that separates the model's signal from the strategy's mechanics

---

## 📊 Key Results

Data: SMH daily prices, 2000-06-05 to 2026-07-24 (6,573 trading days). Test period: late 2022 to July 2026.

- **Model:** XGBoost (Baseline), selected on validation Class-1 F1 (0.560). Test Class-1 F1: 0.612, with accuracy 0.53. A majority-class baseline that always predicts "up/flat" scores higher on both (F1 0.741, accuracy 0.59).
- **GARCH ablation:** removing `garch_vol` lowers mean validation F1 from 0.550 to 0.542, but the gain is positive in only 3 of 5 seeds, so it sits within noise.
- **Confidence tiers:** precision at P ≥ 0.60 / 0.65 / 0.70 is 0.539 / 0.504 / 0.562, all below the 0.578 base rate. Higher confidence does not mean higher accuracy.
- **Simple timing backtest:** ROI of 194.64% vs. 193.49% for the fixed schedule (+1.15 points). The edge is marginal.
- **Tiered backtest:** ROI of 234.92% vs. 212.22% for the fixed schedule. The random-timing benchmark gives a median of 235.70% with no signal at all, so the outperformance comes from the strategy's mechanics (spending more of each year's budget earlier in a rising market), not from the model (p = 0.593).

Bottom line: the pipeline is built to avoid leakage and test-set reuse, and under that discipline the model does not show a reliable timing edge on SMH over this test period.

---

## 🗂 Project Structure

- `main_notebook.ipynb` — Full pipeline: data collection, feature engineering, model training/selection, and backtesting

---

## ⚙️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · CatBoost · Arch (GARCH) · Seaborn · Matplotlib

---

## ▶️ How to Run

\```bash
pip install pandas numpy scipy scikit-learn xgboost catboost arch statsmodels matplotlib seaborn plotly yfinance
\```

1. Open `main_notebook.ipynb` in Jupyter Notebook or VS Code. Data is downloaded from Yahoo Finance and pinned to the original analysis window by `END_DATE` (set it to `None` to use all data available today). If Yahoo is unavailable, place an `smh.csv` file (Date index + Open/High/Low/Close/Volume, adjusted prices) next to the notebook and it will be read instead.
2. Run all cells **in order, top to bottom** (Kernel → Restart & Run All) — several steps (GARCH volatility estimation, VIF-based feature selection, model comparison) depend on earlier cells running in sequence, and partial re-runs can produce inconsistent results.
3. Review the confidence-tier check and backtest results near the end of the notebook before drawing conclusions from the final metrics.

Note: the top candidate models are separated by only a few hundredths of F1, so different library versions can change which model is selected and shift the numbers above. The saved results were produced with Python 3.14, scikit-learn 1.9.1 and yfinance 1.8.0.
