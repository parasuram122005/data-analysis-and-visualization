
import pandas as pd
import numpy as np

def clean_data(df):
    original_shape = df.shape
    print(f"\n--- Data Cleaning ---")

    # Drop full duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed : {before - len(df)}")

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled NaN in '{col}' with median={median_val:.2f}")

    # Fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled NaN in '{col}' with mode='{mode_val}'")

    print(f"Final shape        : {df.shape[0]} rows x {df.shape[1]} columns")
    return df

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    print(f"Outliers removed from '{column}': {before - len(df)}")
    return df

def normalize(df, columns):
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    return df
