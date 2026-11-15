
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Numerical columns
NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]


# Categorical columns
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# Load dataset
def load_dataset(path="data/raw/heart_disease.csv"):

    df = pd.read_csv(path)

    # Check required columns
    required = ALL_FEATURES + ["num"]

    missing_columns = set(required) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Convert columns into numeric format
    for column in required:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Remove rows only if target is missing
    df = df.dropna(subset=["num"]).copy()

    # Check target values
    if not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Unexpected target values!")

    # Convert original target into binary target
    df["target"] = (df["num"] > 0).astype(int)

    # Separate input and output
    X = df[ALL_FEATURES].copy()
    y = df["target"].copy()

    return X, y


# Create preprocessing pipeline
def create_preprocessor():

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    return preprocessor


# Main program
if __name__ == "__main__":

    X, y = load_dataset()

    print("=" * 50)
    print("CARDIOGUARD AI - PREPROCESSING CHECK")
    print("=" * 50)

    print("\nInput Features:", X.shape)
    print("Target Shape:", y.shape)

    print("\nTarget Distribution:")
    print(y.value_counts().sort_index())

    print("\nMissing Values Before Imputation:")
    print(X.isnull().sum())

    print("\nPreprocessor:")
    print(create_preprocessor())

    print("\nPREPROCESSING CONFIGURATION SUCCESSFUL!")
