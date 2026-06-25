# 🇮🇳 Indian Stock Predictor

> AI-powered swing trading signals for NSE listed stocks
> Built by Rishi | SPIT Mumbai | Summer 2026

---

## 📊 Model Performance

| Version | Week | Model | Accuracy |
|---------|------|-------|----------|
| v0.1 | Week 3 | Price chart only | N/A |
| v0.2 | Week 4 | Random Forest | 52.08% |

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `week1_buildtask.ipynb` | Python basics — stock watchlist program |
| `week2_buildtask.ipynb` | OOP — StockPortfolio class |
| `week3_buildtask.py` | Pandas + yfinance — Suzlon price chart |
| `week4_random_forest.py` | First ML model — Random Forest predictor |
| `suzlon_analysis.png` | Suzlon 1-year price chart with MA20/MA50 |

---

## 🛠️ Tech Stack

`Python 3.11` `yfinance` `Pandas` `NumPy` 
`Matplotlib` `TA` `Scikit-learn`

---

## 🔬 Week 4 Notes

- Tested 2y, 3y, 5y data periods systematically
- 3y gave best accuracy (52.08%) for Suzlon
- 5y hurt accuracy — pre-2022 Suzlon was a different company
- Features: RSI, MACD, Price vs MA20, Returns, Volume ratio

---

## 🗺️ Roadmap

- [x] Week 1 — Python basics
- [x] Week 2 — OOP
- [x] Week 3 — Data analysis + first chart
- [x] Week 4 — First ML model (Random Forest)
- [ ] Week 5 — XGBoost + feature engineering
- [ ] Week 6 — LSTM neural network
- [ ] Week 7 — News sentiment (FinBERT)
- [ ] Week 8 — Streamlit deployment

---

## ⚠️ Disclaimer

For educational purposes only. Not financial advice.
Always use stop losses. Past performance ≠ future results.