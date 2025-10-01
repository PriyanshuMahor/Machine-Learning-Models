# Breast Cancer Prediction using Naive Bayes

## Overview

This project demonstrates a machine learning approach to predict whether a breast cancer tumor is **benign** or **malignant** based on cell characteristics. It utilizes the classic Wisconsin Breast Cancer dataset and implements a **Gaussian Naive Bayes** classifier to make predictions.

- **Benign** -  Non-cancerous, typically less aggressive and not life-threatening.
- **Malignant** - Cancerous, potentially invasive and dangerous if untreated.

The notebook covers the complete workflow, including data preprocessing, exploratory data analysis (EDA), feature selection, model training, and evaluation.

-----

## Dataset

The project uses the **Wisconsin Breast Cancer dataset**, sourced from the UCI Machine Learning Repository.

  - The dataset contains **699 samples** and **10 features** that describe characteristics of cell nuclei, plus a sample code number.
  - The target variable, `Class`, is used to classify tumors as either benign or malignant.

**Features:**

1.  Clump Thickness
2.  Uniformity of Cell Size
3.  Uniformity of Cell Shape
4.  Marginal Adhesion
5.  Single Epithelial Cell Size
6.  Bare Nuclei
7.  Bland Chromatin
8.  Normal Nucleoli
9.  Mitoses

-----

## Project Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading & Cleaning**:

      - The `breast_cancer_wisconsin.csv` dataset is loaded into a pandas DataFrame.
      - The `Sample_code_number` column is dropped as it is an identifier and not useful for prediction.
      - Missing values in the `Bare_Nuclei` column are handled using **median imputation**.

2.  **Data Preprocessing**:

      - The `Bare_Nuclei` column is converted to an integer data type.
      - The target variable `Class` is mapped from numeric values (`2`, `4`) to descriptive labels (`Benign`, `Malignant`).

3.  **Exploratory Data Analysis (EDA)**:

      - **Box plots** are generated to visualize the distribution of each feature against the target class. This revealed that most features, especially `Clump_Thickness`, `Uniformity_of_Cell_Size`, and `Bare_Nuclei`, have significantly higher median values for malignant cases.
      - A **correlation heatmap** is created to identify relationships between features. Strong positive correlations were found, notably between `Uniformity of Cell Size` and `Uniformity of Cell Shape` (0.91), indicating potential multicollinearity.

4.  **Feature Selection**:

      - Based on the insights from EDA, four of the most predictive features were manually selected for the model to improve performance and reduce complexity:
          - `Uniformity_of_Cell_Size`
          - `Uniformity_of_Cell_Shape`
          - `Bare_Nuclei`
          - `Clump_Thickness`

5.  **Model Training**:

      - The data is split into training (80%) and testing (20%) sets.
      - A **Gaussian Naive Bayes** model is trained on the selected features.

6.  **Model Evaluation**:

      - The trained model is used to make predictions on the test set.
      - Performance is evaluated using accuracy, a confusion matrix, and a detailed classification report.

-----

## Results

The Gaussian Naive Bayes classifier performed exceptionally well on the test set, demonstrating its effectiveness for this classification task.

  - **Accuracy Score**: **96.43%**

  - **Confusion Matrix**:

    ```
    [[92  3]
     [ 2 43]]
    ```

    This shows that out of 140 test samples, the model misclassified only 5 cases (3 false positives and 2 false negatives).

  - **Classification Report**:

    ```
                  precision    recall  f1-score   support

          Benign       0.98      0.97      0.97        95
       Malignant       0.93      0.96      0.95        45

        accuracy                           0.96       140
       macro avg       0.96      0.96      0.96       140
    weighted avg       0.96      0.96      0.96       140
    ```

The model shows high precision and recall for both classes, indicating a robust and reliable classifier.

## Technologies Used

  - **Python 3**
  - **Pandas** for data manipulation and analysis.
  - **Matplotlib & Seaborn** for data visualization.
  - **Scikit-learn** for model training and evaluation.
  - **Jupyter Notebook** for code execution and presentation.

