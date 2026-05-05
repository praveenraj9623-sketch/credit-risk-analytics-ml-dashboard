import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Credit Portfolio Risk Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Credit Portfolio Risk Analytics Dashboard")
st.write(
    "End-to-end credit risk analytics using PySpark, Databricks SQL, Machine Learning, "
    "and statistical testing."
)

# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/credit_dashboard_data.csv")


@st.cache_data
def load_ml_predictions():
    try:
        return pd.read_csv("data/processed/ml_predictions.csv"), True
    except FileNotFoundError:
        return pd.DataFrame(), False


df = load_data()
ml_df, ml_available = load_ml_predictions()

# ===============================
# SIDEBAR FILTERS
# ===============================

st.sidebar.header("🔍 Filters")

income_filter = st.sidebar.multiselect(
    "Income Segment",
    sorted(df["INCOME_SEGMENT"].dropna().unique()),
    default=sorted(df["INCOME_SEGMENT"].dropna().unique())
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    sorted(df["CODE_GENDER"].dropna().unique()),
    default=sorted(df["CODE_GENDER"].dropna().unique())
)

credit_filter = st.sidebar.multiselect(
    "Credit Risk Band",
    sorted(df["CREDIT_RISK_BAND"].dropna().unique()),
    default=sorted(df["CREDIT_RISK_BAND"].dropna().unique())
)

annuity_filter = st.sidebar.multiselect(
    "Repayment Risk Band",
    sorted(df["ANNUITY_RISK_BAND"].dropna().unique()),
    default=sorted(df["ANNUITY_RISK_BAND"].dropna().unique())
)

filtered_df = df[
    (df["INCOME_SEGMENT"].isin(income_filter)) &
    (df["CODE_GENDER"].isin(gender_filter)) &
    (df["CREDIT_RISK_BAND"].isin(credit_filter)) &
    (df["ANNUITY_RISK_BAND"].isin(annuity_filter))
]

# ===============================
# HELPER FUNCTION
# ===============================

def default_rate_chart(data, group_col):
    chart_df = (
        data.groupby(group_col)["TARGET"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"TARGET": "Default Rate (%)"})
        .sort_values(by="Default Rate (%)", ascending=False)
    )
    return chart_df


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

    total_customers = len(filtered_df)
    default_customers = int(filtered_df["TARGET"].sum())
    default_rate = round(filtered_df["TARGET"].mean() * 100, 2)
    avg_credit = round(filtered_df["AMT_CREDIT"].mean(), 2)

    col1.metric("Total Customers", total_customers)
    col2.metric("Default Customers", default_customers)
    col3.metric("Default Rate (%)", default_rate)
    col4.metric("Average Credit", avg_credit)

    st.divider()

    st.subheader("📊 Default Distribution")

    default_dist = (
        filtered_df["TARGET"]
        .value_counts()
        .rename(index={0: "Non-Default", 1: "Default"})
        .reset_index()
    )

    default_dist.columns = ["Customer Status", "Count"]
    st.bar_chart(default_dist.set_index("Customer Status"))

    st.subheader("💡 Business Summary")

    st.write("""
    - This dashboard analyzes applicant-level credit default risk across a large customer portfolio.
    - The portfolio has an overall default rate of around **8%**, meaning the target class is highly imbalanced.
    - Risk is analyzed using income segment, credit burden, repayment burden, education, and occupation.
    - Machine learning is used as a **baseline risk-screening system**, not as an automatic loan approval system.
    """)

# ===============================
# TAB 2: RISK SEGMENTS
# ===============================

with tab2:
    st.subheader("📊 Default Risk by Income Segment")

    income_risk = default_rate_chart(filtered_df, "INCOME_SEGMENT")
    st.bar_chart(income_risk.set_index("INCOME_SEGMENT"))

    st.write("""
    **Insight:** Medium-income customers show slightly higher default risk in segment-level analysis.
    However, raw income alone was not statistically significant in hypothesis testing.
    """)

    st.subheader("📊 Default Risk by Credit Burden")

    credit_risk = default_rate_chart(filtered_df, "CREDIT_RISK_BAND")
    st.bar_chart(credit_risk.set_index("CREDIT_RISK_BAND"))

    st.write("""
    **Insight:** Credit burden helps identify customers with higher repayment pressure.
    Customers with higher credit-to-income ratios may require closer review.
    """)

    st.subheader("📊 Default Risk by Repayment Burden")

    annuity_risk = default_rate_chart(filtered_df, "ANNUITY_RISK_BAND")
    st.bar_chart(annuity_risk.set_index("ANNUITY_RISK_BAND"))

    st.write("""
    **Insight:** Repayment burden captures how much of the customer’s income is committed to loan repayment.
    This is useful for identifying financially stressed customers.
    """)

    st.subheader("📊 Default Risk by Education Type")

    education_risk = default_rate_chart(filtered_df, "NAME_EDUCATION_TYPE")
    st.bar_chart(education_risk.set_index("NAME_EDUCATION_TYPE"))

    st.write("""
    **Insight:** Education type shows meaningful differences in default risk and is statistically associated
    with default behavior based on the chi-square test.
    """)

    st.subheader("📊 Default Risk by Occupation Type")

    occupation_risk = (
        default_rate_chart(filtered_df, "OCCUPATION_TYPE")
        .head(15)
    )

    st.bar_chart(occupation_risk.set_index("OCCUPATION_TYPE"))

    st.write("""
    **Insight:** Occupation is one of the strongest segment-level differentiators.
    Low-skill laborers show the highest default risk among occupation groups.
    """)

    st.subheader("📄 View Filtered Data")

    if st.checkbox("Show Filtered Data"):
        st.dataframe(filtered_df.head(100), use_container_width=True)

# ===============================
# TAB 3: ML MODEL RESULTS
# ===============================

with tab3:
    st.subheader("🤖 Machine Learning Prediction Results")

    if ml_available:
        col1, col2, col3 = st.columns(3)

        prediction_records = len(ml_df)
        predicted_defaulters = int(ml_df["Predicted_TARGET"].sum())
        avg_probability = round(ml_df["Default_Probability"].mean() * 100, 2)

        col1.metric("Prediction Records", prediction_records)
        col2.metric("Predicted Defaulters", predicted_defaulters)
        col3.metric("Avg Default Probability (%)", avg_probability)

        st.info(
            "This model is used as a baseline risk-screening system. "
            "It helps identify customers who may require further review, not automatic loan rejection."
        )

        st.subheader("📊 Predicted Default Distribution")

        pred_dist = (
            ml_df["Predicted_TARGET"]
            .value_counts()
            .rename(index={0: "Predicted Non-Default", 1: "Predicted Default"})
            .reset_index()
        )

        pred_dist.columns = ["Prediction Status", "Count"]
        st.bar_chart(pred_dist.set_index("Prediction Status"))

        st.subheader("⚠️ High-Risk Customers")

        high_risk = (
            ml_df[ml_df["Default_Probability"] >= 0.55]
            .sort_values(by="Default_Probability", ascending=False)
            .copy()
        )

        high_risk["Default Probability (%)"] = (
            high_risk["Default_Probability"] * 100
        ).round(2)

        preferred_cols = [
            "SK_ID_CURR",
            "Actual_TARGET",
            "Predicted_TARGET",
            "Default Probability (%)",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AGE_YEARS",
            "EMPLOYMENT_YEARS",
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO"
        ]

        available_cols = [col for col in preferred_cols if col in high_risk.columns]

        if len(high_risk) > 0:
            st.dataframe(high_risk[available_cols].head(50), use_container_width=True)
        else:
            st.warning("No high-risk customers found at the current threshold.")

        st.subheader("📌 Model Interpretation")

        st.write("""
        - Logistic Regression was used as a baseline model.
        - Random Forest was selected as the final model because it performed better overall.
        - The dataset is highly imbalanced, with default customers forming a small percentage of the portfolio.
        - Therefore, the model was evaluated using Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
        - Precision is low due to class imbalance, but recall helps identify potentially risky customers.
        - The model is useful for **early risk screening** and portfolio monitoring.
        """)

        st.subheader("📊 Final Model Performance")

        model_metrics = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            "Value": ["~75%", "~14%", "~41%", "~0.21", "~0.65"]
        })

        st.table(model_metrics)

    else:
        st.warning(
            "ML predictions file not found. Please create `data/processed/ml_predictions.csv` first."
        )

# ===============================
# TAB 4: HYPOTHESIS TESTING
# ===============================

with tab4:
    st.subheader("🧪 Statistical Hypothesis Testing")

    st.write("""
    These tests validate whether key variables are statistically related to default risk.
    A p-value below 0.05 means the relationship is statistically significant.
    """)

    # ============================
    # 1. Income T-test
    # ============================

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
        st.info(
            "Result: No statistically significant income difference was found. "
            "Income segmentation is useful for business analysis, but raw income alone is not a strong statistical differentiator."
        )

    # ============================
    # 2. Credit Amount T-test
    # ============================

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
        st.success(
            "Result: Credit amount is statistically different between default and non-default customers."
        )
    else:
        st.info("Result: No significant credit amount difference found.")

    # ============================
    # 3. Education Chi-square
    # ============================

    education_table = pd.crosstab(df["NAME_EDUCATION_TYPE"], df["TARGET"])
    chi2_edu, p_edu, dof_edu, expected_edu = chi2_contingency(education_table)

    st.markdown("### 3. Education Type vs Default Risk")
    st.write("H0: Education type is not associated with default risk.")
    st.write("H1: Education type is associated with default risk.")
    st.write("P-value:", p_edu)

    if p_edu < 0.05:
        st.success(
            "Result: Education type is significantly associated with default risk."
        )
    else:
        st.info("Result: No significant association found.")

    # ============================
    # 4. Occupation Chi-square
    # ============================

    occupation_table = pd.crosstab(df["OCCUPATION_TYPE"], df["TARGET"])
    chi2_occ, p_occ, dof_occ, expected_occ = chi2_contingency(occupation_table)

    st.markdown("### 4. Occupation vs Default Risk")
    st.write("H0: Occupation is not associated with default risk.")
    st.write("H1: Occupation is associated with default risk.")
    st.write("P-value:", p_occ)

    if p_occ < 0.05:
        st.success(
            "Result: Occupation is significantly associated with default risk."
        )
    else:
        st.info("Result: No significant association found.")

    st.subheader("📌 Summary of Hypothesis Testing")

    hypothesis_summary = pd.DataFrame({
        "Test": [
            "Income vs Default",
            "Credit Amount vs Default",
            "Education vs Default",
            "Occupation vs Default"
        ],
        "Method": [
            "T-test",
            "T-test",
            "Chi-square",
            "Chi-square"
        ],
        "Result": [
            "Not statistically significant" if p_income >= 0.05 else "Statistically significant",
            "Statistically significant" if p_credit < 0.05 else "Not statistically significant",
            "Statistically significant" if p_edu < 0.05 else "Not statistically significant",
            "Statistically significant" if p_occ < 0.05 else "Not statistically significant"
        ]
    })

    st.table(hypothesis_summary)