import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

st.set_page_config(
    page_title="Credit Portfolio Risk Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Credit Portfolio Risk Analytics Dashboard")
st.write("End-to-end credit risk analytics using PySpark, Databricks SQL, Machine Learning, and statistical testing.")

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("data/processed/credit_dashboard_data.csv")

# Try loading ML predictions
try:
    ml_df = pd.read_csv("data/processed/ml_predictions.csv")
    ml_available = True
except:
    ml_available = False

# ===============================
# SIDEBAR FILTERS
# ===============================

st.sidebar.header("🔍 Filters")

income_filter = st.sidebar.multiselect(
    "Income Segment",
    df["INCOME_SEGMENT"].unique(),
    default=df["INCOME_SEGMENT"].unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    df["CODE_GENDER"].unique(),
    default=df["CODE_GENDER"].unique()
)

credit_filter = st.sidebar.multiselect(
    "Credit Risk Band",
    df["CREDIT_RISK_BAND"].unique(),
    default=df["CREDIT_RISK_BAND"].unique()
)

annuity_filter = st.sidebar.multiselect(
    "Repayment Risk Band",
    df["ANNUITY_RISK_BAND"].unique(),
    default=df["ANNUITY_RISK_BAND"].unique()
)

filtered_df = df[
    (df["INCOME_SEGMENT"].isin(income_filter)) &
    (df["CODE_GENDER"].isin(gender_filter)) &
    (df["CREDIT_RISK_BAND"].isin(credit_filter)) &
    (df["ANNUITY_RISK_BAND"].isin(annuity_filter))
]

# ===============================
# TABS
# ===============================

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Portfolio Overview",
    "📊 Risk Segments",
    "🤖 ML Model Results",
    "🧪 Hypothesis Testing"
])

# ===============================
# TAB 1: PORTFOLIO OVERVIEW
# ===============================

with tab1:
    st.subheader("📌 Key Portfolio Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(filtered_df))
    col2.metric("Default Customers", int(filtered_df["TARGET"].sum()))
    col3.metric("Default Rate (%)", round(filtered_df["TARGET"].mean() * 100, 2))
    col4.metric("Average Credit", round(filtered_df["AMT_CREDIT"].mean(), 2))

    st.divider()

    st.subheader("📊 Default Distribution")

    default_dist = filtered_df["TARGET"].value_counts().rename(index={
        0: "Non-Default",
        1: "Default"
    })

    st.bar_chart(default_dist)

    st.subheader("💡 Business Summary")

    st.write("""
    - This dashboard analyzes applicant-level credit risk patterns.
    - The target variable identifies customers with repayment difficulty.
    - Risk is analyzed using income, credit burden, repayment burden, education, and occupation.
    - Machine learning is used to support early default-risk identification.
    """)

# ===============================
# TAB 2: RISK SEGMENTS
# ===============================

with tab2:
    st.subheader("📊 Default Risk by Income Segment")

    income_risk = (
        filtered_df.groupby("INCOME_SEGMENT")["TARGET"]
        .mean()
        .reset_index()
        .sort_values(by="TARGET", ascending=False)
    )

    st.bar_chart(income_risk.set_index("INCOME_SEGMENT"))

    st.subheader("📊 Default Risk by Credit Burden")

    credit_risk = (
        filtered_df.groupby("CREDIT_RISK_BAND")["TARGET"]
        .mean()
        .reset_index()
        .sort_values(by="TARGET", ascending=False)
    )

    st.bar_chart(credit_risk.set_index("CREDIT_RISK_BAND"))

    st.subheader("📊 Default Risk by Repayment Burden")

    annuity_risk = (
        filtered_df.groupby("ANNUITY_RISK_BAND")["TARGET"]
        .mean()
        .reset_index()
        .sort_values(by="TARGET", ascending=False)
    )

    st.bar_chart(annuity_risk.set_index("ANNUITY_RISK_BAND"))

    st.subheader("📊 Default Risk by Education Type")

    education_risk = (
        filtered_df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
        .mean()
        .reset_index()
        .sort_values(by="TARGET", ascending=False)
    )

    st.bar_chart(education_risk.set_index("NAME_EDUCATION_TYPE"))

    st.subheader("📊 Default Risk by Occupation Type")

    occupation_risk = (
        filtered_df.groupby("OCCUPATION_TYPE")["TARGET"]
        .mean()
        .reset_index()
        .sort_values(by="TARGET", ascending=False)
        .head(15)
    )

    st.bar_chart(occupation_risk.set_index("OCCUPATION_TYPE"))

    st.subheader("📄 View Filtered Data")

    if st.checkbox("Show Filtered Data"):
        st.dataframe(filtered_df.head(100))

# ===============================
# TAB 3: ML MODEL RESULTS
# ===============================

with tab3:
    st.subheader("🤖 Machine Learning Prediction Results")

    if ml_available:
        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction Records", len(ml_df))
        col2.metric("Predicted Defaulters", int(ml_df["Predicted_TARGET"].sum()))
        col3.metric(
            "Avg Default Probability",
            round(ml_df["Default_Probability"].mean() * 100, 2)
        )

        st.subheader("📊 Predicted Default Distribution")

        pred_dist = ml_df["Predicted_TARGET"].value_counts().rename(index={
            0: "Predicted Non-Default",
            1: "Predicted Default"
        })

        st.bar_chart(pred_dist)

        st.subheader("⚠️ High-Risk Customers")

        high_risk = ml_df[
            ml_df["Default_Probability"] >= 0.55
        ].sort_values(by="Default_Probability", ascending=False)

        st.dataframe(high_risk.head(50))

        st.subheader("📌 Model Interpretation")

        st.write("""
        - Random Forest was selected as the final model.
        - Threshold tuning identified 0.55 as the best threshold based on F1-score.
        - The model showed moderate ROC-AUC and is useful as a baseline risk-screening system.
        - Precision is low due to class imbalance, but recall helps identify potential risky customers.
        """)

    else:
        st.warning("ML predictions file not found. Please create `data/processed/ml_predictions.csv` first.")

# ===============================
# TAB 4: HYPOTHESIS TESTING
# ===============================

with tab4:
    st.subheader("🧪 Statistical Hypothesis Testing")

    st.write("These tests validate whether key variables are statistically related to default risk.")

    # Income T-test
    safe_income = df[df["TARGET"] == 0]["AMT_INCOME_TOTAL"]
    risk_income = df[df["TARGET"] == 1]["AMT_INCOME_TOTAL"]

    t_stat_income, p_income = ttest_ind(
        safe_income,
        risk_income,
        equal_var=False
    )

    st.markdown("### 1. Income vs Default Risk")
    st.write("H0: Income is not significantly different between default and non-default customers.")
    st.write("H1: Income is significantly different between default and non-default customers.")
    st.write("P-value:", p_income)

    if p_income < 0.05:
        st.success("Result: Income is significantly different between the two groups.")
    else:
        st.info("Result: No significant income difference found.")

    # Credit amount T-test
    safe_credit = df[df["TARGET"] == 0]["AMT_CREDIT"]
    risk_credit = df[df["TARGET"] == 1]["AMT_CREDIT"]

    t_stat_credit, p_credit = ttest_ind(
        safe_credit,
        risk_credit,
        equal_var=False
    )

    st.markdown("### 2. Credit Amount vs Default Risk")
    st.write("H0: Credit amount is not significantly different between default and non-default customers.")
    st.write("H1: Credit amount is significantly different between default and non-default customers.")
    st.write("P-value:", p_credit)

    if p_credit < 0.05:
        st.success("Result: Credit amount is significantly different between the two groups.")
    else:
        st.info("Result: No significant credit amount difference found.")

    # Education Chi-square
    education_table = pd.crosstab(df["NAME_EDUCATION_TYPE"], df["TARGET"])
    chi2_edu, p_edu, dof_edu, expected_edu = chi2_contingency(education_table)

    st.markdown("### 3. Education Type vs Default Risk")
    st.write("H0: Education type is not associated with default risk.")
    st.write("H1: Education type is associated with default risk.")
    st.write("P-value:", p_edu)

    if p_edu < 0.05:
        st.success("Result: Education type is significantly associated with default risk.")
    else:
        st.info("Result: No significant association found.")

    # Occupation Chi-square
    occupation_table = pd.crosstab(df["OCCUPATION_TYPE"], df["TARGET"])
    chi2_occ, p_occ, dof_occ, expected_occ = chi2_contingency(occupation_table)

    st.markdown("### 4. Occupation vs Default Risk")
    st.write("H0: Occupation is not associated with default risk.")
    st.write("H1: Occupation is associated with default risk.")
    st.write("P-value:", p_occ)

    if p_occ < 0.05:
        st.success("Result: Occupation is significantly associated with default risk.")
    else:
        st.info("Result: No significant association found.")