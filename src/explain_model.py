
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import issparse

from sklearn.model_selection import train_test_split

from preprocessing import load_dataset


print("=" * 60)
print("       CARDIOGUARD AI - SHAP EXPLAINABILITY")
print("=" * 60)


# 1. Create output directory
output_dir = Path("reports/shap")
output_dir.mkdir(parents=True, exist_ok=True)


# 2. Load dataset
X, y = load_dataset()

print("\nDataset loaded successfully!")
print("Dataset shape:", X.shape)


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Load best trained model
model = joblib.load("models/best_model.pkl")

print("\nBest model loaded successfully!")


# 5. Extract preprocessing and classifier
preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]


# 6. Transform data using fitted preprocessor
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Convert sparse matrices to dense arrays if necessary
if issparse(X_train_transformed):
    X_train_transformed = X_train_transformed.toarray()

if issparse(X_test_transformed):
    X_test_transformed = X_test_transformed.toarray()


# 7. Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

X_train_transformed = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
)

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
)

print("\nTransformed features:", len(feature_names))


# 8. Select background and explanation samples
background = X_train_transformed.sample(
    n=min(50, len(X_train_transformed)),
    random_state=42
)

explain_samples = X_test_transformed.head(30)


# 9. Create SHAP explainer
print("\nCreating SHAP explainer...")

explainer = shap.Explainer(
    classifier,
    background
)

shap_values = explainer(explain_samples)

print("SHAP values generated successfully!")


# 10. SHAP summary plot
plt.figure(figsize=(11, 7))

shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)

plt.title("SHAP Feature Impact Summary")
plt.tight_layout()

plt.savefig(
    output_dir / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 11. SHAP feature importance bar plot
plt.figure(figsize=(11, 7))

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    output_dir / "shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 12. Individual prediction explanation
plt.figure(figsize=(12, 7))

shap.plots.waterfall(
    shap_values[0],
    max_display=15,
    show=False
)

plt.title("Individual Prediction Explanation")

plt.tight_layout()

plt.savefig(
    output_dir / "shap_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 13. Save feature importance values
importance = np.abs(
    shap_values.values
).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_SHAP_Importance": importance
})

importance_df = importance_df.sort_values(
    by="Mean_SHAP_Importance",
    ascending=False
)

importance_df.to_csv(
    output_dir / "shap_feature_importance.csv",
    index=False
)


# 14. Display results
print("\nTOP 10 IMPORTANT FEATURES")

print(
    importance_df.head(10).to_string(index=False)
)

print("\nGenerated files:")

for file in sorted(output_dir.iterdir()):
    print("-", file.name)

print("\nSHAP EXPLAINABILITY COMPLETED!")