# STOCK RISK & FORECAST DASHBOARD PIPELINE 

import os
import glob
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DATA_FOLDER = "data"
OUTPUT_MASTER = os.path.join(DATA_FOLDER, "stocks_master_enriched.csv")
OUTPUT_SUMMARY = os.path.join(DATA_FOLDER, "stock_summary.csv")
RISK_FREE_RATE_ANNUAL = 0.03
TRADING_DAYS = 252
ROLL_WINDOW = 21

#LOAD & COMBINE CSV FILES ----------
all_files = glob.glob(os.path.join(DATA_FOLDER, "Financial Data - *.csv"))

df_list = []
for file in all_files:
    ticker = os.path.basename(file).replace("Financial Data - ", "").replace(".csv", "")
    temp = pd.read_csv(file)
    temp["Ticker"] = ticker
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# DAILY RETURNS
df["Return"] = df.groupby("Ticker")["Close"].pct_change()
df["CumulativeReturn"] = (1 + df["Return"]).groupby(df["Ticker"]).cumprod() - 1

# Rolling Volatility (21 days) & Annualized Vol
df["RollingVol_21D"] = (
    df.groupby("Ticker")["Return"]
    .rolling(window=ROLL_WINDOW)
    .std()
    .reset_index(level=0, drop=True)
)
df["RollingVol_21D_Ann"] = df["RollingVol_21D"] * math.sqrt(TRADING_DAYS)

# DAILY RETURNS
df["Return"] = df.groupby("Ticker")["Close"].pct_change()
df["CumulativeReturn"] = (1 + df["Return"]).groupby(df["Ticker"]).cumprod() - 1
df["RollingVol_21D"] = (
    df.groupby("Ticker")["Return"]
    .rolling(window=ROLL_WINDOW)
    .std()
    .reset_index(level=0, drop=True)
)
df["RollingVol_21D_Ann"] = df["RollingVol_21D"] * math.sqrt(TRADING_DAYS)

# Rolling 21-day Sharpe Ratio
df["RollingSharpe_21D"] = (
    df.groupby("Ticker")["Return"]
    .rolling(21)
    .apply(lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() != 0 else np.nan)
    .reset_index(level=0, drop=True)
)

# Drawdown
df["RunningMax"] = df.groupby("Ticker")["Close"].cummax()
df["Drawdown"] = (df["Close"] / df["RunningMax"]) - 1


# BENCHMARK 
market_returns = df.groupby("Date")["Return"].mean().reset_index().rename(columns={"Return": "MarketReturn"})
df = df.merge(market_returns, on="Date", how="left")

#HELPERS
def annualize_return(daily_ret):
    return (1 + daily_ret) ** TRADING_DAYS - 1

def annualize_vol(daily_vol):
    return daily_vol * math.sqrt(TRADING_DAYS)

def sharpe_ratio(returns, rf_annual=RISK_FREE_RATE_ANNUAL):
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    rf_daily = rf_annual / TRADING_DAYS
    excess = r - rf_daily
    mu = excess.mean()
    sd = excess.std()
    if sd == 0 or pd.isna(sd):
        return np.nan
    return (mu / sd) * math.sqrt(TRADING_DAYS)

def beta_calc(asset_ret, market_ret):
    df_temp = pd.concat([asset_ret, market_ret], axis=1).dropna()
    df_temp.columns = ["ri", "rm"]
    if len(df_temp) < 20:
        return np.nan
    var_m = df_temp["rm"].var()
    if var_m == 0 or pd.isna(var_m):
        return np.nan
    cov = df_temp["ri"].cov(df_temp["rm"])
    return cov / var_m

def forecast_returns(returns, horizon=21):
    r = returns.dropna()
    if len(r) < 20:
        return np.nan
    X = np.arange(len(r)).reshape(-1, 1)
    model = LinearRegression().fit(X, r.values)
    future_idx = np.arange(len(r), len(r) + horizon).reshape(-1, 1)
    preds = model.predict(future_idx)
    return np.mean(preds)

# PER-TICKER METRICS
summary_rows = []
for ticker in df["Ticker"].unique():
    sub = df[df["Ticker"] == ticker].copy()
    r = sub["Return"].dropna()

    avg_daily = r.mean()
    ann_ret = annualize_return(avg_daily)
    daily_vol = r.std()
    ann_vol = annualize_vol(daily_vol)
    sharpe = sharpe_ratio(r)
    beta = beta_calc(sub["Return"], sub["MarketReturn"])
    total_ret = sub["CumulativeReturn"].dropna().iloc[-1] if not sub["CumulativeReturn"].dropna().empty else np.nan

    pred_mean = forecast_returns(r, 21)
    pred_cum = (1 + pred_mean) ** 21 - 1 if not pd.isna(pred_mean) else np.nan
    pred_ann = annualize_return(pred_mean) if not pd.isna(pred_mean) else np.nan

    summary_rows.append({
        "Ticker": ticker,
        "AvgDailyReturn": avg_daily,
        "AnnReturn": ann_ret,
        "DailyVol": daily_vol,
        "AnnVol": ann_vol,
        "SharpeRatio": sharpe,
        "Beta": beta,
        "TotalReturn": total_ret,
        "PredMeanDaily": pred_mean,
        "PredCum21D": pred_cum,
        "PredAnnReturn": pred_ann
    })

summary = pd.DataFrame(summary_rows)

#OUTPUT FILES
df.to_csv(OUTPUT_MASTER, index=False)
summary.to_csv(OUTPUT_SUMMARY, index=False)

print("\n✅ Success! Files created:")
print(f"  → {OUTPUT_MASTER}")
print(f"  → {OUTPUT_SUMMARY}")
print("\nNow import these CSVs into Power BI for your dashboard!")

