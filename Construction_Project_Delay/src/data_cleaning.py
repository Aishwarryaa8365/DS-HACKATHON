import pandas as pd
import os

def load_and_clean_data(filepath):
    """Load CSV, validate, clean, and engineer the delay target column."""
    df = pd.read_csv(filepath)

    print("=== DATA CLEANING ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Drop duplicates if any
    df.drop_duplicates(inplace=True)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Validate numeric ranges
    assert df['Planned_Progress'].between(0, 100).all(), "Planned_Progress out of range"
    assert df['Actual_Progress'].between(0, 100).all(), "Actual_Progress out of range"
    assert (df['Budget'] > 0).all(), "Budget must be positive"
    assert (df['Workforce'] > 0).all(), "Workforce must be positive"

    # ---------------------------------------------------------------
    # TARGET ENGINEERING RULE (documented):
    # Progress_Gap = Planned_Progress - Actual_Progress
    # A project is labelled Delayed (1) if Progress_Gap > 10 percentage
    # points, meaning actual progress lags planned by more than 10 pp.
    # Otherwise it is labelled On Track (0).
    # ---------------------------------------------------------------
    df['Progress_Gap'] = df['Planned_Progress'] - df['Actual_Progress']
    df['Delayed'] = (df['Progress_Gap'] > 10).astype(int)

    print(f"\nDelay target distribution:\n{df['Delayed'].value_counts()}")
    print(f"\nProgress_Gap stats:\n{df['Progress_Gap'].describe()}")

    # Save cleaned data
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'reports')
    os.makedirs(out_dir, exist_ok=True)
    cleaned_path = os.path.join(out_dir, 'cleaned_data.csv')
    df.to_csv(cleaned_path, index=False)
    print(f"\nCleaned data saved to: {cleaned_path}")

    return df
