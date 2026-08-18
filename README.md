# Machine Learning Assignment 2

## a. Problem Statement

Build and evaluate multiple classification models on a public classification dataset and
demonstrate the trained models through an interactive Streamlit application.

The application accepts test data in CSV format, allows the user to select a classification
model, and displays Accuracy, AUC, Precision, Recall, F1 Score, MCC, a confusion matrix,
a classification report, and prediction output.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository  
**DOI:** 10.24432/C5DW2B

The dataset contains 569 instances and 30 real-valued features. The target is binary:
Benign (0) and Malignant (1). The features describe characteristics computed from
digitized images of fine needle aspirates of breast masses.

The dataset satisfies the assignment requirement of at least 12 features and 500 instances.

## c. GitHub Repository Link

**GitHub Repository:** TO BE ADDED AFTER CREATING THE REPOSITORY

## d. Models Used

The assignment document lists five specific classification models. All five are implemented
on the same dataset and evaluated using the required six metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9737 | 0.9944 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy=0.9649, AUC=0.9960, Precision=0.9750, Recall=0.9286, F1=0.9512, MCC=0.9245. |
| Decision Tree | Accuracy=0.9211, AUC=0.9448, Precision=0.9459, Recall=0.8333, F1=0.8861, MCC=0.8299. |
| kNN | Accuracy=0.9561, AUC=0.9825, Precision=0.9744, Recall=0.9048, F1=0.9383, MCC=0.9058. |
| Naive Bayes | Accuracy=0.9386, AUC=0.9934, Precision=1.0000, Recall=0.8333, F1=0.9091, MCC=0.8715. |
| Random Forest | Accuracy=0.9737, AUC=0.9944, Precision=1.0000, Recall=0.9286, F1=0.9630, MCC=0.9442. |
| **Overall Winner** | **Random Forest** achieved the strongest overall balance according to the F1 score, with AUC, accuracy and MCC also considered. |

## Methodology

1. Load the Breast Cancer Wisconsin (Diagnostic) dataset.
2. Use a stratified 80:20 train-test split with `random_state=42`.
3. Standardize features for Logistic Regression and kNN.
4. Train Decision Tree, Gaussian Naive Bayes and Random Forest directly on the feature values.
5. Evaluate every model on the same held-out test set.
6. Save the trained models as Joblib files.
7. Use the saved models in the Streamlit application.

## Repository Structure

```text
ML_Assignment_2_Final/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── test_data_sample.csv
├── model_metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## Streamlit Application

**Live Streamlit App:** TO BE ADDED AFTER DEPLOYMENT

The application provides:
- CSV test-data upload
- Model selection dropdown
- Evaluation metrics
- Confusion matrix
- Classification report
- Prediction output download

## Reproducibility

The train/test split uses `random_state=42`. Random Forest also uses `random_state=42`.
The test file used for the reported results is included as `test_data.csv`.

## Dataset Citation

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.
https://doi.org/10.24432/C5DW2B.
