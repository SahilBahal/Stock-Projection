import pandas as pd

df = pd.read_csv("data/stocks_master_enriched.csv")

# Pivot to wide format
pivot = df.pivot(index="Date", columns="Ticker", values="Return")

# Compute correlation matrix
corr = pivot.corr()

# Save as CSV
corr.to_csv("data/correlation_matrix.csv")
print("✅ Created data/correlation_matrix.csv for Power BI heatmap.")
