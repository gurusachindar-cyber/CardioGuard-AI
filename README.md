# CardioGuard-AI
Heart Disease Prediction Using Machine Learning
## Model Training and Evaluation Results

Five machine learning classification algorithms were trained and evaluated for the CardioGuard-AI heart disease prediction project.

The dataset was divided into 80% training data and 20% testing data using stratified sampling. The models were evaluated using Accuracy, Precision, Recall, F1-score and ROC-AUC.

### Performance Comparison

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 88.52% | 83.87% | 92.86% | 88.14% | 96.65% |
| SVM | 88.52% | 83.87% | 92.86% | 88.14% | 96.43% |
| KNN | 88.52% | 86.21% | 89.29% | 87.72% | 95.29% |
| Random Forest | 86.89% | 83.33% | 89.29% | 86.21% | 94.48% |
| Decision Tree | 78.69% | 74.19% | 82.14% | 77.97% | 85.98% |

### Best Model

Logistic Regression achieved the highest ROC-AUC score of 96.65% among the five evaluated algorithms. Therefore, it was selected as the best-performing model based on ROC-AUC.

### Discussion

The results indicate that Logistic Regression and SVM achieved the highest accuracy of 88.52%. Logistic Regression obtained the highest ROC-AUC of 96.65%, while KNN achieved the highest precision of 86.21%.

Decision Tree obtained the lowest performance across the reported metrics. These results show that different algorithms have different classification characteristics.

The evaluation results are based on the project's 80:20 train-test split with random state 42. They represent model performance on the selected test dataset and do not guarantee the same performance on new data.
