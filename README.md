# Rain Prediction using Machine Learning

This project is part of a lab assignment to build and evaluate a machine learning model that predicts whether it will rain today based on weather features. The entire pipeline includes data preprocessing, model training, hyperparameter tuning, evaluation, and feature importance analysis.

## 🚀 Objective
To develop a classification model using Random Forest and Logistic Regression to predict the `RainToday` target variable using various weather-related features.

## 📊 Dataset
The dataset contains historical weather data, including features like temperature, humidity, wind speed, and cloud cover. The target variable is `RainToday`, indicating whether it rained or not.

## 🛠️ Steps Performed

### 1. Data Splitting
- The dataset was split into training and test sets using `train_test_split` with stratification to maintain class balance.

### 2. Preprocessing
- Numerical and categorical features were detected automatically using `select_dtypes`.
- A `ColumnTransformer` was created with:
  - StandardScaler for numerical features.
  - OneHotEncoder for categorical features.

### 3. Pipeline Creation
- A pipeline was built by combining the preprocessor with:
  - `RandomForestClassifier`
  - Later, replaced with `LogisticRegression` for comparison.

### 4. Model Tuning with GridSearchCV
- Performed hyperparameter tuning using GridSearchCV with accuracy scoring and cross-validation.

### 5. Evaluation
- Test set accuracy was reported.
- Classification report and confusion matrix were generated.
- Feature importance was extracted from the trained Random Forest model.

### 6. Visualization
- Bar chart showing the top feature importances.
- Confusion matrix heatmap plotted for model evaluation.

## 🔍 Results
- **Best model accuracy** on test set: *Reported in output cell.*
- **Most important feature** for prediction: `Rainfall`

## 📁 Files in this Repository
- `RainPrediction_ProjectLab.ipynb` – Complete Jupyter Notebook with code and outputs
- `README.md` – This file

## 📦 Requirements
Install dependencies:
```bash
pip install pandas scikit-learn matplotlib seaborn
