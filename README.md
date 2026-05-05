# 📊 Credit Portfolio Risk Analytics

An end-to-end Data Science project to analyze and predict customer default risk using PySpark, Databricks, Machine Learning, and Streamlit.

---

## 🔗 Live Demo

👉 Streamlit App:  
https://credit-risk-analytics-ml-dashboard-aebrncwcarrlfhvkfzkjnq.streamlit.app/

👉 Databricks Dashboard:  
https://dbc-f4b685c9-52f9.cloud.databricks.com/sql/dashboardsv3/01f146e4dfcb165fa31c89e27f2e5b28?scrollToWidget=01f146e5c99714289120fe5b054e641c&o=7474659916375521

> Note: Databricks dashboard access may require login. Screenshots are included in the repository.


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

- The portfolio contains more than 300K customers with an overall default rate of around 8.07%.
- The dataset is highly imbalanced, so model evaluation focused on Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix instead of accuracy alone.
- Medium-income customers show slightly higher default risk in segment-level analysis.
- Raw income difference between default and non-default customers was not statistically significant based on t-test.
- Credit amount showed statistically significant variation between default and non-default customers.
- Education type and occupation type are significantly associated with default risk based on chi-square testing.
- Low-skill laborers show the highest occupation-level default risk.
- Credit burden and repayment burden features help identify financially stressed customers.

## 🧪 Hypothesis Testing Results

- Income vs Default Risk: Not statistically significant.
- Credit Amount vs Default Risk: Statistically significant.
- Education Type vs Default Risk: Statistically significant association.
- Occupation Type vs Default Risk: Statistically significant association.
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
