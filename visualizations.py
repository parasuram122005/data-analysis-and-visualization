
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

def plot_distributions(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n = len(numeric_cols)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        df[col].dropna().hist(ax=ax, bins=20, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Distribution of {col}", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    plt.suptitle("Feature Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("distributions.png", dpi=150, bbox_inches="tight")
    print("  Saved: distributions.png")
    plt.close()

def plot_heatmap(corr):
    plt.figure(figsize=(8, 6))
    mask = corr.isnull()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, mask=mask,
        annot_kws={"size": 10}
    )
    plt.title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("heatmap.png", dpi=150, bbox_inches="tight")
    print("  Saved: heatmap.png")
    plt.close()

def plot_boxplots(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n = len(numeric_cols)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols):
        df[[col]].dropna().boxplot(ax=ax, patch_artist=True,
            boxprops=dict(facecolor="#4C72B0", color="#333"),
            medianprops=dict(color="white", linewidth=2))
        ax.set_title(f"Box Plot: {col}", fontsize=12)
    plt.suptitle("Outlier Detection via Box Plots", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("boxplots.png", dpi=150, bbox_inches="tight")
    print("  Saved: boxplots.png")
    plt.close()

def plot_scatter(df, x_col, y_col):
    plt.figure(figsize=(7, 5))
    sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    plt.title(f"{x_col} vs {y_col}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"scatter_{x_col}_vs_{y_col}.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: scatter_{x_col}_vs_{y_col}.png")
    plt.close()
