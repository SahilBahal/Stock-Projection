import pandas as pd

# Load enriched data
df = pd.read_csv("data/stocks_master_enriched.csv")

# Calculate average daily return across all tickers
portfolio = df.groupby("Date")["Return"].mean().reset_index()
portfolio["CumReturn"] = (1 + portfolio["Return"]).cumprod() - 1

# Save for Power BI
portfolio.to_csv("data/portfolio.csv", index=False)
print("✅ Created data/portfolio.csv for Power BI portfolio analysis.")
