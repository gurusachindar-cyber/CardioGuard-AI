
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

from preprocessing import load_dataset


print("=" * 60)
print("    CARDIOGUARD AI - MODEL EVALUATION")
print("=" * 60)

# Create output directory
output_dir = Path("reports/model_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Load dataset
X, y = load_dataset()

# Same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load model comparison
results = pd.read_csv(
    output_dir / "model_comparison.csv"
)

model_names = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
    "KNN"
]

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl",
    "KNN": "knn.pkl"
}

trained_models = {}

for name, filename in model_files.items():
    trained_models[name] = joblib.load(
        Path("models") / filename
    )

sns.set_theme(style="whitegrid")


# 1. Model comparison chart
plt.figure(figsize=(10, 6))

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.title("Machine Learning Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("ML Algorithms")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(output_dir / "accuracy_comparison.png")
plt.close()


# 2. All performance metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

metric_data = results.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=metric_data,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Performance Metrics Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(output_dir / "metrics_comparison.png")
plt.close()


# 3. Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for ax, name in zip(axes.flat, model_names):

    model = trained_models[name]
    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

axes.flat[5].axis("off")

plt.suptitle("Confusion Matrix Comparison", fontsize=16)
plt.tight_layout()

plt.savefig(output_dir / "confusion_matrices.png")
plt.close()


# 4. ROC curves
plt.figure(figsize=(9, 7))

for name in model_names:

    model = trained_models[name]

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(output_dir / "roc_curve_comparison.png")
plt.close()


# 5. Save detailed classification reports
with open(
    output_dir / "classification_reports.txt",
    "w"
) as file:

    for name in model_names:

        predictions = trained_models[name].predict(X_test)

        file.write("\n" + "=" * 50 + "\n")
        file.write(name + "\n")
        file.write("=" * 50 + "\n")

        file.write(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )


# Final output
print("\nMODEL EVALUATION COMPLETED!")

print("\nGenerated files:")

for file in sorted(output_dir.iterdir()):
    print("-", file.name)

print("\nAll evaluation results saved successfully!")
