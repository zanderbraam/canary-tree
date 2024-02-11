import os

import pandas as pd
# import matplotlib.pyplot as plt

path_to_raw_data = "./data/raw"

# Load data
etf_data = pd.read_csv(os.path.join(path_to_raw_data, "etfs_adj_close.csv"), index_col=0)

strategy_data = pd.read_csv(
    os.path.join(path_to_raw_data, "aiam_nas30_sp30_monthly_long_10_short_0_long_only_no_canary_performance.csv"),
    index_col=0)

strategy_data = strategy_data.loc[:, "equity"].to_frame().dropna()
strategy_data.columns = ["strategy"]

# Drop empty columns
etf_data = etf_data.dropna(axis=1, how="all")

# Drop certain columns
etf_data = etf_data.drop(
    ["BITO", "ARKK", "EFNL", "GREK", "ARGT", "AMLP", "EGPT", "EIRL", "EPI", "EPOL", "GXG", "NORW", "THD", "TUR", "UBT",
     "UST", "VXX"], axis=1)

# Drop columns that have unchangeing values
unique_counts = etf_data.nunique()
columns_to_drop = unique_counts[unique_counts == 1].index
etf_data = etf_data.drop(columns=columns_to_drop)

# Forward fill data
etf_data = etf_data.ffill()
etf_data = etf_data.reindex(strategy_data.index)
etf_data = etf_data.ffill()

# SANITY CHECK
# Find first index with no more nans
# na_check = etf_data.isna()
# rows_with_no_nans = ~na_check.any(axis=1)
# first_index_no_nans = rows_with_no_nans.idxmax()
# print(first_index_no_nans)
#
# na_check = economic_data.isna()
# rows_with_no_nans = ~na_check.any(axis=1)
# first_index_no_nans = rows_with_no_nans.idxmax()
# print(first_index_no_nans)
#
# economic_data.to_csv("economics.csv")
# etf_data.to_csv("etfs.csv")
# strategy_data.to_csv("strategy.csv")

# CLEAN DATA

# Convert the index to datetime
etf_data.index = pd.to_datetime(etf_data.index)

# Drop the last row if it's not the last business day of the month
if etf_data.index[-1] != etf_data.index[-1].to_period("M").to_timestamp("M"):
    etf_data = etf_data.iloc[:-1]

# Resample to get the last business day of each month
monthly_etf_data = etf_data.resample("BM").last()

# Compute percent change for each period
one_month = monthly_etf_data.pct_change(1).fillna(0.0)
three_months = monthly_etf_data.pct_change(3).fillna(0.0)
six_months = monthly_etf_data.pct_change(6).fillna(0.0)
twelve_months = monthly_etf_data.pct_change(12).fillna(0.0)

# Compute weighted momentum
result = ((one_month * 12) + (three_months * 4) + (six_months * 2) + twelve_months) / 4.0

# Convert the index back to string
result.index = result.index.strftime("%Y-%m-%d")

# Convert the date column to datetime format
strategy_data.index = pd.to_datetime(strategy_data.index)

# Resample the data at the end of each business month and take the last observation as the representative value
monthly_strategy_data = strategy_data.resample("BM").last()

# Calculate monthly returns and then transform into binary variables (1 if the return is positive, 0 otherwise)
monthly_returns = monthly_strategy_data.pct_change()
binary_returns = (monthly_returns > 0).astype(int)
binary_returns = binary_returns.shift(-1)
binary_returns = binary_returns.iloc[:-1]

result = result.iloc[1:]
result = result.iloc[:-1]
binary_returns = binary_returns[1:]

result.to_csv("./data/transformed/input_data_etfs_only.csv")
binary_returns.to_csv("./data/transformed/target_data_etfs_only.csv")
