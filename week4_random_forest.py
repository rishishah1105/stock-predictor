# Week 4 Build Task — Random Forest Prediction Model
# Stock: Suzlon Energy (SUZLON.NS)
# Period: 3y (best accuracy after testing 2y/3y/5y)
# Accuracy: 52.08%
# v0.2 milestone
 
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = yf.download("SUZLON.NS", period="3y", progress=False, auto_adjust=True)
data.columns = data.columns.get_level_values(0)
# .copy() prevents SettingWithCopyWarning
data = data.copy()
print(f"Downloaded {len(data)} days of Suzlon data")
print(data.tail(3))

# RSI — measures overbought/oversold conditions
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
# MACD — measures momentum direction
macd = ta.trend.MACD(data['Close'])
data['MACD'] = macd.macd()
# Price vs 20-day moving average
data['Price_vs_MA20'] = data['Close'] / data['Close'].rolling(20).mean()
# 1-day return — yesterday momentum
data['Return_1d'] = data['Close'].pct_change(1)
# 5-day return — weekly momentum
data['Return_5d'] = data['Close'].pct_change(5)
# Volume ratio — is today's volume unusual?
# Above 1.0 = higher than average = something is happening
data['Volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()

print(data[['Close', 'RSI', 'MACD', 'Price_vs_MA20']].tail(3))

# Target: Will price go UP tomorrow?
# shift(-1) looks at TOMORROW's closing price
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

print(f"Total usable rows after dropna: {len(data)}")
print(f"UP days: {data['Target'].sum()}")
print(f"DOWN days: {len(data) - data['Target'].sum()}")

features = ['Return_1d', 'Return_5d', 'Volume_ratio', 'RSI', 'MACD', 'Price_vs_MA20']

X = data[features].values
y = data['Target'].values

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully")

accuracy = model.score(X_test, y_test)
predictions = model.predict(X_test)

print(f"✅ Directional Accuracy: {accuracy:.2%}")
print("\n📊 Detailed Report:")
print(classification_report(y_test, predictions, target_names=['DOWN ↓', 'UP ↑']))

# Feature importance tells you WHICH features
# the model found most useful for predictions
# Higher number = more useful signal
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔍 What the model finds most useful:")
print(importance.to_string(index=False))

# Take the most recent row — that's today's data
latest = data[features].iloc[-1].values.reshape(1, -1)

# Predict direction
prediction = model.predict(latest)[0]

# predict_proba gives confidence percentage
# [0] = DOWN probability, [1] = UP probability
confidence = model.predict_proba(latest)[0]

direction = "UP ↑" if prediction == 1 else "DOWN ↓"
print(f"\n🔮 Suzlon Prediction for Tomorrow:")
print(f"Direction:  {direction}")
print(f"Confidence: {confidence[1]:.1%} UP / {confidence[0]:.1%} DOWN")