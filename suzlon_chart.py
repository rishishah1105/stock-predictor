import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

print("📥 Downloading Suzlon Energy data...")
data = yf.download("SUZLON.NS", period="1y")

data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Chart 1: Price + Moving Averages
axes[0].plot(data['Close'], label='Price', color='#2196F3', linewidth=1.5)
axes[0].plot(data['MA20'], label='20-day MA', color='#FF9800', linewidth=1)
axes[0].plot(data['MA50'], label='50-day MA', color='#F44336', linewidth=1)
axes[0].set_title('Suzlon Energy — Price Chart', fontsize=14)
axes[0].set_ylabel('Price (₹)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Chart 2: Volume
axes[1].bar(data.index, data['Volume'].squeeze(), color='#9C27B0', alpha=0.6)
axes[1].set_title('Trading Volume')
axes[1].set_ylabel('Volume')
axes[1].grid(True, alpha=0.3)

# Chart 3: Cumulative Returns
axes[2].plot(data['Cumulative_Return'] * 100, color='#4CAF50', linewidth=1.5)
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=0.8)
axes[2].set_title('Cumulative Return (%)')
axes[2].set_ylabel('Return %')
axes[2].grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('suzlon_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary stats
current = float(data['Close'].iloc[-1].squeeze())
high_52 = float(data['High'].max().squeeze())
low_52 = float(data['Low'].min().squeeze())
avg_volume = float(data['Volume'].mean().squeeze())
total_return = float(data['Cumulative_Return'].iloc[-1].squeeze() * 100)

print(f"\n📊 Suzlon Energy — 1 Year Summary")
print(f"{'='*35}")
print(f"Current Price:    ₹{current:.2f}")
print(f"52-Week High:     ₹{high_52:.2f}")
print(f"52-Week Low:      ₹{low_52:.2f}")
print(f"Avg Daily Volume: {avg_volume:,.0f}")
print(f"1-Year Return:    {total_return:.1f}%")