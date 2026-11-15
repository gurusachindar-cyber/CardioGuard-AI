
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("=" * 60)
print("       CARDIOGUARD AI - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Folder setup
data_path = Path("data/raw/heart_disease.csv")
output_folder = Path("reports/eda")
output_folder.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Feature information
feature_info = {
    "age": "Age in years",
    "sex": "Sex (0/1)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar category",
    "restecg": "Resting ECG category",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels",
    "thal": "Thalassemia category",
    "num": "Original target (0-4)"
}

# 1. Basic information
print("\n1. DATASET INFORMATION")
print("-" * 60)
print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# 2. Statistical summary
print("\n2. STATISTICAL SUMMARY")
print("-" * 60)
print(df.describe().round(2))

# 3. Missing values
print("\n3. MISSING VALUE ANALYSIS")
print("-" * 60)

missing = df.isnull().sum()
print(missing)

print("\nMissing percentage:")
print((missing / len(df) * 100).round(2))

# 4. Duplicate records
print("\n4. DUPLICATE ANALYSIS")
print("-" * 60)
print("Duplicate records:", df.duplicated().sum())

# 5. Target distribution
print("\n5. TARGET DISTRIBUTION")
print("-" * 60)
print(df["num"].value_counts().sort_index())

# 6. Unique values
print("\n6. UNIQUE VALUES")
print("-" * 60)

for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")

# 7. Feature descriptions
print("\n7. FEATURE DESCRIPTIONS")
print("-" * 60)

for column, description in feature_info.items():
    print(f"{column}: {description}")

# Plot style
sns.set_theme(style="whitegrid")

# 8. Target distribution chart
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="num")
plt.title("Original Heart Disease Target Distribution")
plt.xlabel("Target Category")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(output_folder / "target_distribution.png")
plt.show()

# 9. Missing values chart
plt.figure(figsize=(9, 5))
missing[missing > 0].plot(kind="bar")
plt.title("Missing Values by Feature")
plt.xlabel("Feature")
plt.ylabel("Missing Count")
plt.tight_layout()
plt.savefig(output_folder / "missing_values.png")
plt.show()

# 10. Numerical feature distributions
numerical_columns = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

df[numerical_columns].hist(
    figsize=(12, 9),
    bins=15,
    edgecolor="black"
)

plt.suptitle("Numerical Feature Distributions")
plt.tight_layout()
plt.savefig(output_folder / "feature_distributions.png")
plt.show()

# 11. Correlation heatmap
plt.figure(figsize=(12, 8))
correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(output_folder / "correlation_heatmap.png")
plt.show()

# 12. Outlier investigation
plt.figure(figsize=(12, 7))

sns.boxplot(data=df[numerical_columns], orient="h")

plt.title("Numerical Feature Outlier Investigation")
plt.tight_layout()
plt.savefig(output_folder / "outlier_analysis.png")
plt.show()

# 13. Feature relationship with target
plt.figure(figsize=(10, 6))

sns.boxplot(data=df, x="num", y="age")

plt.title("Age Distribution Across Target Categories")
plt.xlabel("Original Target Category")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(output_folder / "age_vs_target.png")
plt.show()

# 14. Save EDA summary
summary = pd.DataFrame({
    "Feature": df.columns,
    "Data_Type": df.dtypes.astype(str).values,
    "Missing_Count": df.isnull().sum().values,
    "Unique_Values": df.nunique().values
})

summary.to_csv(
    output_folder / "eda_summary.csv",
    index=False
)

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")
for file in output_folder.iterdir():
    print(file)

print("\nAll EDA results saved inside reports/eda/")
