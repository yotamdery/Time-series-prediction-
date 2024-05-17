import pandas as pd


def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Group by 'month' and aggregate the values
    aggregated_df = df.groupby('month').agg({
        'spend': lambda x: x.mean() if not x.isna().all() else float('nan'),
        'revenue': lambda x: x.mean() if not x.isna().all() else float('nan'),
        'subs': lambda x: x.mean() if not x.isna().all() else float('nan')
    }).reset_index()

    return aggregated_df