
from pathlib import Path
import pandas as pd
import requests
from io import StringIO

print("=" * 55)
print("       CARDIOGUARD AI - DATASET DOWNLOADER")
print("=" * 55)

url = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

try:
    print("\nDownloading official UCI dataset...")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.text),
        header=None,
        names=columns,
        na_values="?"
    )

    output_folder = Path("data/raw")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_folder / "heart_disease.csv"
    df.to_csv(output_file, index=False)

    print("\nDOWNLOAD SUCCESSFUL!")
    print("-" * 55)

    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df["num"].value_counts().sort_index())

    print("\nSaved at:", output_file.resolve())

except Exception as error:
    print("\nDOWNLOAD FAILED!")
    print("Reason:", error)
