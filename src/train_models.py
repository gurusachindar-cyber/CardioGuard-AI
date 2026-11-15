
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from preprocessing import load_dataset, create_preprocessor


print("=" * 60)
print("       CARDIOGUARD AI - MODEL TRAINING")
print("=" * 60)


# 1. Load dataset
X, y = load_dataset()

print("\nDataset Loaded Successfully!")
print("Features:", X.shape)
print("Target:", y.shape)


# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# 3. Define five ML algorithms
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}


# 4. Create folders
Path("models").mkdir(exist_ok=True)
Path("reports/model_results").mkdir(
    parents=True,
    exist_ok=True
)


# 5. Train and evaluate
results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    # Preprocessing + model pipeline
    pipeline = Pipeline([
        ("preprocessing", create_preprocessor()),
        ("classifier", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, y_probability
    )

    # Display results
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = pipeline

    # Save trained model
    filename = name.lower().replace(" ", "_")
    joblib.dump(
        pipeline,
        f"models/{filename}.pkl"
    )


# 6. Compare all models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("       MODEL COMPARISON RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

# Save comparison
results_df.to_csv(
    "reports/model_results/model_comparison.csv",
    index=False
)


# 7. Select best model using ROC-AUC
best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest model by test ROC-AUC:", best_model_name)

print("\nAll models saved successfully!")
print("Comparison report saved successfully!")

print("\nMODEL TRAINING COMPLETED!")
