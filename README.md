# CardioGuard-AI – Heart Disease Prediction Using Machine Learning

## 1. Project Overview
CardioGuard-AI is a machine learning-based application designed to predict the possibility of heart disease using patient health information. It uses multiple machine learning algorithms and compares their performance to identify the best-performing model.

## 2. Algorithms Used
This project uses five machine learning algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

## 3. Dataset and Preprocessing
The dataset is stored at `data/raw/heart_disease.csv`.

Preprocessing techniques include:
- Handling missing numerical values using median imputation.
- Handling missing categorical values using most-frequent imputation.
- Scaling numerical features using StandardScaler.
- Encoding categorical features using OneHotEncoder.
- Converting the original target into a binary classification target.

## 4. Model Evaluation Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 88.52% | 83.87% | 92.86% | 88.14% | 96.65% |
| SVM | 88.52% | 83.87% | 92.86% | 88.14% | 96.43% |
| KNN | 88.52% | 86.21% | 89.29% | 87.72% | 95.29% |
| Random Forest | 86.89% | 83.33% | 89.29% | 86.21% | 94.48% |
| Decision Tree | 78.69% | 74.19% | 82.14% | 77.97% | 85.98% |

**Best-performing model:** Logistic Regression, with a ROC-AUC score of 96.65%.

The models were evaluated using an 80:20 stratified train-test split.

## 5. Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

## 6. Project Structure

```text
CardioGuard-AI/
│
├── app.py
├── src/
├── data/
│   └── raw/
│       └── heart_disease.csv
├── models/
├── notebooks/
├── pages/
├── assets/
├── reports/
├── screenshots/
└── requirements.txt
```

## 7. Installation and Execution

**Step 1: Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**Step 2: Open the project folder**

```bash
cd CardioGuard-AI
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Run the Streamlit application**

```bash
streamlit run app.py
```

## 8. Application Screenshots

Screenshots of the CardioGuard-AI application are available in the `screenshots/` folder.

## 9. Deployment

Live Application URL: Add your deployed Streamlit URL here.

GitHub Repository: Add your GitHub repository URL here.

## 10. Disclaimer

This application is developed for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
