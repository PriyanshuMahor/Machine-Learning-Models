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
      - Outliers are handled using the capping (or Winsorizing) method. For each feature, values falling outside 1.5 times the Interquartile Range (IQR) are replaced with the respective upper or lower bound. This technique mitigates the effect of extreme values without causing data loss.

3.  **Exploratory Data Analysis (EDA)**:

      - Box plots are used to visualize the distribution of features for Benign vs. Malignant cases on the capped data, showing that malignant tumors generally have higher feature values.
      - <img width="1489" height="1490" alt="image" src="https://github.com/user-attachments/assets/38ac8712-cd05-4b3d-849f-038e82b837fd" />

      - A correlation heatmap reveals strong positive correlations between features like `Uniformity of Cell Size` and `Uniformity of Cell Shape`, indicating multicollinearity.
      - <img width="933" height="850" alt="image" src="https://github.com/user-attachments/assets/81e75a7f-eccc-4b4e-802c-cdc0db4276c4" />


4.  **Feature Selection**:

      - Based on the insights from EDA, four of the most predictive features were manually selected for the model to improve performance and reduce complexity:
          - `Uniformity_of_Cell_Size`
          - `Uniformity_of_Cell_Shape`
          - `Bare_Nuclei`
          - `Clump_Thickness`

5.  **Model Training**:

      - The data is split into training (80%) and testing (20%) sets. Crucially, `stratify=Y` is used to ensure that the proportion of benign and malignant samples is the same in both the train and test sets.
      - A **Gaussian Naive Bayes** model is trained on the selected features.

6.  **Model Evaluation**:

      - The trained model is used to make predictions on the test set.
      - Performance is evaluated using accuracy, a confusion matrix, and a detailed classification report.

-----

## Results

The Gaussian Naive Bayes classifier performed-

  - **Accuracy Score**: **95.71%**

  - **Confusion Matrix**:

    ```
    [[89  3]
     [ 3 45]]
    ```

    This shows that out of 140 test samples, the model misclassified only 5 cases (3 false positives and 2 false negatives).

  - **Classification Report**:

    ```
                  precision    recall  f1-score   support

      Benign       0.97      0.97      0.97        92
     Malignant       0.94      0.94      0.94        48

    accuracy                           0.96       140
     macro avg       0.95      0.95      0.95       140
    weighted avg       0.96      0.96      0.96       140
    ```

The model shows high precision and recall for both classes, indicating a robust and reliable classifier.

## Technologies Used

  - **Pandas** for data manipulation and analysis.
  - **Matplotlib & Seaborn** for data visualization.
  - **Scikit-learn** for model training and evaluation.

