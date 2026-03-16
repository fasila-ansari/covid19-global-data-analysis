# ============================================================
# COVID-19 Global Data Analysis
# Data Science Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("===================================")
print("COVID-19 Global Data Analysis Start")
print("===================================")

# ============================================================
# Load Dataset
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv("owid-covid-data.csv")

print("\nDataset Loaded Successfully")
print("\nDataset Shape:", df.shape)

print("\nAvailable Columns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# Select Important Columns (SAFE VERSION)
# ============================================================

required_columns = [
    "location",
    "date",
    "total_cases",
    "new_cases",
    "total_deaths",
    "new_deaths",
    "population"
]

df = df[required_columns]

# Convert date column
df["date"] = pd.to_datetime(df["date"])

print("\nFiltered Dataset Shape:", df.shape)

# ============================================================
# Data Cleaning
# ============================================================

print("\nHandling missing values...")

df = df.dropna(subset=["total_cases", "population"])

df.fillna(0, inplace=True)

print("Missing values handled.")

# ============================================================
# Global Case Trend Over Time
# ============================================================

print("\nGenerating global cases over time visualization...")

global_cases = df.groupby("date")["new_cases"].sum()

plt.figure(figsize=(10,5))

plt.plot(global_cases)

plt.title("Global COVID-19 New Cases Over Time")

plt.xlabel("Date")

plt.ylabel("New Cases")

plt.savefig("cases_over_time.png")

plt.show()

# ============================================================
# Top 10 Countries by Total Cases
# ============================================================

print("\nAnalyzing top affected countries...")

latest_data = df.sort_values("date").groupby("location").tail(1)

top_countries = latest_data.sort_values(
    by="total_cases",
    ascending=False
).head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x="total_cases",
    y="location",
    data=top_countries
)

plt.title("Top 10 Countries by Total COVID-19 Cases")

plt.xlabel("Total Cases")

plt.ylabel("Country")

plt.savefig("top_countries_cases.png")

plt.show()

# ============================================================
# Death Rate Analysis
# ============================================================

print("\nCalculating mortality rates...")

latest_data["death_rate"] = (
    latest_data["total_deaths"] / latest_data["total_cases"]
)

death_rate = latest_data.sort_values(
    by="death_rate",
    ascending=False
).head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x="death_rate",
    y="location",
    data=death_rate
)

plt.title("Top Countries by COVID-19 Mortality Rate")

plt.xlabel("Death Rate")

plt.ylabel("Country")

plt.savefig("death_rate.png")

plt.show()

# ============================================================
# Correlation Analysis
# ============================================================

print("\nGenerating correlation heatmap...")

numeric_cols = df.select_dtypes(include=["float64","int64"])

corr = numeric_cols.corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=False
)

plt.title("COVID-19 Data Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()

# ============================================================
# Key Insights
# ============================================================

print("\nKey Insights:")

print("1. Global COVID-19 cases increased rapidly during major outbreak waves.")
print("2. A small number of countries contributed significantly to total case counts.")
print("3. Mortality rates varied significantly across countries.")
print("4. Pandemic spread showed strong correlation between cases and deaths.")

print("\nCOVID-19 Data Analysis Completed Successfully!")