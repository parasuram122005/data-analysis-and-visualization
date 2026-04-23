
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaner import clean_data
from visualizations import plot_distributions, plot_heatmap, plot_boxplots

def load_data(filepath="sample_data.csv"):
    df = pd.read_csv(filepath)
    print(f"\n{'='*40}")
    print(f"Dataset loaded: {filepath}")
    print(f"Shape         : {df.shape[0]} rows x {df.shape[1]} columns")
    return df

def summarize(df):
    print(f"\n--- Statistical Summary ---")
    print(df.describe().round(2))
    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values found.")
    print(f"\n--- Data Types ---")
    print(df.dtypes)
    return df

def correlation_analysis(df):
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"\n--- Correlation Matrix ---")
    print(numeric_df.corr().round(2))
    return numeric_df.corr()

def run_eda(filepath="sample_data.csv"):
    df = load_data(filepath)
    df = clean_data(df)
    summarize(df)
    corr = correlation_analysis(df)
    plot_distributions(df)
    plot_heatmap(corr)
    plot_boxplots(df)
    print("\nEDA complete! Check the generated plots.")

if __name__ == "__main__":
    run_eda()
