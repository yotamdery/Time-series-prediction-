import pandas as pd
from statsmodels.tsa.stattools import adfuller


def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Group by 'month' and aggregate the values
    aggregated_df = df.groupby('month').agg({
        'spend': lambda x: x.mean() if not x.isna().all() else float('nan'),
        'revenue': lambda x: x.mean() if not x.isna().all() else float('nan'),
        'subs': lambda x: x.mean() if not x.isna().all() else float('nan')
    }).reset_index()

    return aggregated_df

def perform_adf_test_all_features(df: pd.DataFrame, features: list) -> None:
    print("ADF test results:")
    for feature in features:
        print(f"\nTest results for feature: {feature.capitalize()}")
        result = adfuller(df[feature])

        # Extract and print the ADF test results
        adf_statistic, p_value, used_lag, _, _, _ = result
        print(f'Used lag: {used_lag}')
        print(f'ADF Statistic: {adf_statistic}')
        print(f'p-value: {p_value}')

        # Interpretation
        if p_value < 0.05:
            print("The time series is stationary.")
        else:
            print("The time series is non-stationary.")
        print("##########################################\n")