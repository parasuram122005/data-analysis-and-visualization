# Data Analysis and Visualization

Exploratory Data Analysis (EDA) and visualization of real-world datasets using Python.

## Features
- Load and clean real-world CSV datasets
- Handle missing values and outliers
- Statistical summary and correlation analysis
- 10+ types of visualizations (bar, scatter, heatmap, histogram, box plot)
- Insights extraction from data patterns

## Tech Stack
- Python 3
- Pandas — data manipulation
- NumPy — numerical computing
- Matplotlib — plotting
- Seaborn — statistical visualizations

## Setup
```bash
pip install pandas numpy matplotlib seaborn
```

## How to Run
```bash
python eda_analysis.py
```

## Project Structure
```
data-analysis-and-visualization/
├── eda_analysis.py       # Main EDA script
├── visualizations.py     # All chart functions
├── data_cleaner.py       # Data preprocessing utilities
├── sample_data.csv       # Sample dataset
├── requirements.txt      # Dependencies
└── README.md
```

## Visualizations Included
- Distribution histograms
- Correlation heatmap
- Box plots for outlier detection
- Scatter plots with regression line
- Bar charts for categorical data
- Pie charts for proportions
- Pair plots for multi-variable analysis

## Sample Output
```
Dataset Shape   : (150, 5)
Missing Values  : 0
Duplicate Rows  : 0
Numeric Columns : 4
---------------------------------
Correlation with target:
  feature_1  :  0.87
  feature_2  :  0.72
  feature_3  : -0.45
```
