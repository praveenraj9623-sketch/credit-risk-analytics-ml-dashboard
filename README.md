# 📊 Credit Portfolio Risk Analytics

An end-to-end Data Science project to analyze and predict customer default risk using PySpark, Databricks, Machine Learning, and Streamlit.

---

## 🚀 Project Overview

This project focuses on identifying high-risk customers in a credit portfolio by combining:

* Data Engineering (PySpark)
* Data Analysis (Databricks SQL)
* Machine Learning (Scikit-learn)
* Business Intelligence Dashboard (Streamlit)
* Statistical Validation (Hypothesis Testing)

---

## 🧱 Tech Stack

* Python (Pandas, NumPy)
* PySpark (Data Processing)
* Databricks (SQL Analytics & Dashboard)
* Scikit-learn (ML Models)
* Streamlit (Web App Dashboard)
* SciPy (Hypothesis Testing)

---

## 📁 Project Structure

```
credit-portfolio-risk-analytics/
│
├── app.py                         # Streamlit dashboard
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                      # Original dataset (Kaggle)
│   └── processed/                # Cleaned + ML outputs
│       ├── credit_dashboard_data.csv
│       └── ml_predictions.csv
│
├── notebooks/
│   └── ML_training.ipynb         # ML development notebook
│
├── src/
│   ├── data_cleaning.py          # PySpark cleaning
│   ├── feature_engineering.py    # Feature creation
│   ├── sql_analysis.sql          # Databricks SQL queries
│   └── model_training.py         # ML logic (optional)
│
└── reports/
    └── business_insights.md      # Final insights
```

---

## 📊 Key Features

### 🔹 1. Data Processing (PySpark)

* Missing value handling
* Outlier removal
* Feature engineering (income ratio, age, etc.)

### 🔹 2. Databricks SQL Analytics

* Default rate by income, occupation, education
* Risk segmentation (credit burden, repayment burden)
* Interactive dashboards

### 🔹 3. Machine Learning Models

* Logistic Regression (baseline)
* Random Forest (final model)
* Threshold tuning (optimal = 0.55)

### 🔹 4. Model Evaluation

* Accuracy, Precision, Recall, F1-score
* ROC-AUC analysis
* Confusion Matrix

### 🔹 5. Hypothesis Testing

* T-test (income & credit)
* Chi-square test (education & occupation)

### 🔹 6. Streamlit Dashboard

* Interactive filters
* Risk segmentation charts
* ML prediction insights
* High-risk customer identification

---

## 🤖 Model Performance

| Metric    | Value |
| --------- | ----- |
| Accuracy  | ~75%  |
| Recall    | ~41%  |
| Precision | ~14%  |
| F1-score  | ~0.21 |
| ROC-AUC   | ~0.65 |

👉 Random Forest performed better than Logistic Regression.

---

## 💡 Key Business Insights

* Medium-income customers show higher default risk
* Higher credit burden increases default probability
* Occupation and education significantly impact risk
* Repayment pressure (annuity ratio) is a strong indicator

---

## 🧪 Hypothesis Testing Results

* Income difference between defaulters and non-defaulters is statistically significant
* Credit amount shows significant variation across risk groups
* Education and occupation are strongly associated with default risk

---

## 📌 How to Run the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run Streamlit App

```
streamlit run app.py
```

---

## 📊 Data Source

* Home Credit Default Risk Dataset (Kaggle)

---

## 🧠 Author

**Praveen Raj A**
MCA (Data Science)
Aspiring Data Scientist

---

## ⭐ Project Highlights

✔ End-to-end pipeline
✔ Real-world dataset
✔ ML + Analytics + Dashboard
✔ Production-style structure

---

## 🔥 Future Improvements

* SMOTE for class imbalance
* XGBoost / LightGBM
* Model deployment API
* Real-time prediction UI
