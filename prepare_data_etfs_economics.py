import os

import pandas as pd
# import matplotlib.pyplot as plt

path_to_raw_data = "./data/raw"

# Load data
etf_data = pd.read_csv(os.path.join(path_to_raw_data, "etfs_adj_close.csv"), index_col=0)
economic_data = pd.read_csv(os.path.join(path_to_raw_data, "us_economics_monthly.csv"), index_col=0)

strategy_data = pd.read_csv(
    os.path.join(path_to_raw_data, "aiam_nas30_sp30_monthly_long_10_short_0_long_only_no_canary_performance.csv"),
    index_col=0)

strategy_data = strategy_data.loc[:, "equity"].to_frame().dropna()
strategy_data.columns = ["strategy"]

etf_data = etf_data.loc[:"2022-02-01", :]
economic_data = economic_data.loc[:"2022-02-01", :]
strategy_data = strategy_data.loc[:"2022-02-01", :]

# Drop empty columns
etf_data = etf_data.dropna(axis=1, how="all")
economic_data = economic_data.dropna(axis=1, how="all")

# Drop certain columns
etf_data = etf_data.drop(
    ["BITO", "ARKK", "EFNL", "GREK", "ARGT", "AMLP", "EGPT", "EIRL", "EPI", "EPOL", "GXG", "NORW", "THD", "TUR", "UBT",
     "UST", "VXX"], axis=1)
economic_data = economic_data.drop(
    ["45997", "63496", "70893", "79854", "46321", "43693", "43704", "43728", "43730", "43905", "77430", "46372"],
    axis=1)

# Drop columns that have unchangeing values
unique_counts = etf_data.nunique()
columns_to_drop = unique_counts[unique_counts == 1].index
etf_data = etf_data.drop(columns=columns_to_drop)

unique_counts = economic_data.nunique()
columns_to_drop = unique_counts[unique_counts == 1].index
economic_data = economic_data.drop(columns=columns_to_drop)

# Forward fill data
etf_data = etf_data.ffill()
economic_data = economic_data.reindex(etf_data.index)

for column in economic_data.columns:
    # Find the last valid index for the column
    last_valid_index = economic_data.loc[:, column].last_valid_index()
    if last_valid_index is not None:
        # Identify the range of indices up to the last valid index
        valid_range = economic_data.index <= last_valid_index
        # Perform forward fill on the identified range in a single step
        economic_data.loc[valid_range, column] = economic_data.loc[valid_range, column].ffill()

etf_data = etf_data.reindex(strategy_data.index).ffill()
economic_data = economic_data.reindex(strategy_data.index)

economic_data.index = pd.to_datetime(economic_data.index)
columns_to_drop = []
for column in economic_data.columns:
    # Find the last valid index for the column
    last_valid_date = economic_data[column].last_valid_index()
    if last_valid_date is not None:
        # Calculate the gap in days from the last valid date to the last date in the DataFrame
        gap = (economic_data.index[-1] - last_valid_date).days

        # Check if the gap is more than two months (approximated as 60 days)
        if gap > 60:
            # Mark the column for dropping if the gap is more than two months
            columns_to_drop.append(column)
        else:
            # Forward fill the column if the gap is two months or less
            economic_data.loc[:, column] = economic_data.loc[:, column].ffill()

# Drop the marked columns
economic_data = economic_data.drop(columns=columns_to_drop)
economic_data.index = economic_data.index.strftime('%Y-%m-%d')

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

# Combine data
combined_data = etf_data.join(economic_data, how="inner")
combined_data = combined_data.replace(0.0, 1e-10)

# Convert the index to datetime
combined_data.index = pd.to_datetime(combined_data.index)

# Drop the last row if it's not the last business day of the month
if combined_data.index[-1] != combined_data.index[-1].to_period("M").to_timestamp("M"):
    combined_data = combined_data.iloc[:-1]

# Resample to get the last business day of each month
monthly_combined_data = combined_data.resample("BM").last()

# Compute percent change for each period
one_month = monthly_combined_data.pct_change(1).fillna(0.0)
three_months = monthly_combined_data.pct_change(3).fillna(0.0)
six_months = monthly_combined_data.pct_change(6).fillna(0.0)
twelve_months = monthly_combined_data.pct_change(12).fillna(0.0)

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

definitions = pd.read_csv("./data/raw/economic_indicator_definitions.csv")

data = {}
for series_id in economic_data.columns:
    description = definitions.loc[definitions['series_id'] == int(series_id), 'description'].values[0]
    data[series_id] = description

data = pd.DataFrame.from_dict(data, orient='index', columns=['description'])
data.to_csv("./data/transformed/economic_descriptions.csv")

result.iloc[1:, :].to_csv("./data/transformed/input_data_etfs_economics.csv")
binary_returns.iloc[1:, :].to_csv("./data/transformed/target_data_etfs_economics.csv")
