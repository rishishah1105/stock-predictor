# Week 5 Build Task — XGBoost + Feature Engineering
# Stock: Suzlon Energy (SUZLON.NS)
# Period: 3y (proven best from Week 4 testing)
# Target: 60-63% accuracy (up from 52.08% in v0.2)
# v0.3 milestone

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

data = yf.download("SUZLON.NS", period="3y", progress=False, auto_adjust=True)
# Flatten MultiIndex columns → simple OHLCV
data.columns = data.columns.get_level_values(0)
# .copy() prevents SettingWithCopyWarning
data = data.copy()

print(f"Downloaded {len(data)} days of Suzlon data")


data['Return_1d'] = data['Close'].pct_change(1)
data['Return_5d'] = data['Close'].pct_change(5)
data['Return_10d'] = data['Close'].pct_change(10)

data['Volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
macd = ta.trend.MACD(data['Close'])
data['MACD'] = macd.macd()

data['Price_vs_MA20'] = data['Close'] / data['Close'].rolling(20).mean()
data['Price_vs_MA50'] = data['Close'] / data['Close'].rolling(50).mean()

data['High_Low_range'] = (data['High'] - data['Low']) / data['Close']
data['High_52w'] = data['High'].rolling(252).max()
data['Low_52w'] = data['Low'].rolling(252).min()
data['Position_52w'] = (data['Close'] - data['Low_52w']) / \
(data['High_52w'] - data['Low_52w'])

# ── TARGET COLUMN ─────────────────────────────────────────────────
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

print(f"Total usable rows: {len(data)}")
print(f"UP days:   {data['Target'].sum()}")
print(f"DOWN days: {len(data) - data['Target'].sum()}")

features = [
    'Return_1d', 'Return_5d', 'Volume_ratio',
    'RSI', 'MACD', 'Price_vs_MA20',
    'Return_10d', 'High_Low_range', 'Price_vs_MA50',
    'Position_52w'
]

X = data[features].values
y = data['Target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

xgb = XGBClassifier(
    n_estimators=200,     # 200 trees — more than RF's 100
                          # sequential so doesn't overfit as easily
    max_depth=4,          # tree depth — 4 is sweet spot
                          # too deep = overfitting
                          # too shallow = underfitting
    learning_rate=0.05,   # small steps = more careful learning
                          # 0.05 is conservative, prevents overshooting
    random_state=42,      # reproducible results
    eval_metric='logloss',# measures probability error
                          # better than accuracy for training
    verbosity=0           # silent training, no spam output
)

xgb.fit(X_train, y_train)
print("✅ XGBoost model trained successfully")
accuracy = xgb.score(X_test, y_test)
predictions = xgb.predict(X_test)

print(f"✅ XGBoost Accuracy: {accuracy:.2%}")
print("\n📊 Detailed Report:")
print(classification_report(y_test, predictions, target_names=['DOWN ↓', 'UP ↑']))

# Which of the 10 features does XGBoost find most useful?
importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔍 Feature Importance:")
print(importance.to_string(index=False))

# Take most recent row — today's Suzlon data
latest = data[features].iloc[-1].values.reshape(1, -1)

# Predict direction
prediction = xgb.predict(latest)[0]

# predict_proba gives probability for each class
# [0] = DOWN probability, [1] = UP probability
confidence = xgb.predict_proba(latest)[0]

direction = "UP ↑" if prediction == 1 else "DOWN ↓"
print(f"\n🔮 Suzlon Prediction for Tomorrow:")
print(f"Direction:  {direction}")
print(f"Confidence: {confidence[1]:.1%} UP / {confidence[0]:.1%} DOWN")

# Save trained model to disk
# So we can load it later without retraining
# .pkl = pickle file = frozen model saved to disk
joblib.dump(xgb, 'suzlon_xgb_model.pkl')
print("\n💾 Model saved as suzlon_xgb_model.pkl")

train_accuracy = xgb.score(X_train, y_train)
test_accuracy = xgb.score(X_test, y_test)

print(f"Train accuracy: {train_accuracy:.2%}")
print(f"Test accuracy:  {test_accuracy:.2%}")
print(f"Gap:            {(train_accuracy - test_accuracy):.2%}")