from pyspark.sql import SparkSession

# Step 1: Start Spark
spark = SparkSession.builder \
    .appName("Credit Portfolio Risk Analytics") \
    .getOrCreate()

# Step 2: Load dataset
df = spark.read.csv(
    "data/application_train.csv",
    header=True,
    inferSchema=True
)

# Step 3: Show first 5 rows
df.show(5)

# Step 4: Show column names and data types
df.printSchema()

# Step 5: Count rows and columns
print("Total Rows:", df.count())
print("Total Columns:", len(df.columns))

# Step 6: Check target distribution
df.groupBy("TARGET").count().show()

from pyspark.sql.functions import col, when, count

from pyspark.sql.functions import col, when, count

from pyspark.sql.functions import col, when, count

# Step 7: Missing values (clean format)
missing_data = []

for c in df.columns:
    missing_count = df.filter(col(c).isNull()).count()
    missing_data.append((c, missing_count))

# Convert to Spark DataFrame
missing_df = spark.createDataFrame(missing_data, ["Column", "Missing_Count"])

# Show only columns with missing values
missing_df.filter(col("Missing_Count") > 0).show(20, truncate=False)
# Stop Spark

# Step 8: Drop columns with too many missing values (>50%)

total_rows = df.count()

columns_to_drop = []

for row in missing_df.collect():
    col_name = row["Column"]
    missing = row["Missing_Count"]

    if missing > (0.5 * total_rows):
        columns_to_drop.append(col_name)

print("Columns to drop:", columns_to_drop)

df = df.drop(*columns_to_drop)

print("Remaining columns:", len(df.columns))

# Step 9: Fill missing values

from pyspark.sql.types import StringType, IntegerType, DoubleType, FloatType
from pyspark.sql.functions import col

numeric_cols = []
categorical_cols = []

for field in df.schema.fields:
    if isinstance(field.dataType, (IntegerType, DoubleType, FloatType)):
        numeric_cols.append(field.name)
    elif isinstance(field.dataType, StringType):
        categorical_cols.append(field.name)

# Fill numeric missing values with 0
df = df.fillna(0, subset=numeric_cols)

# Fill categorical missing values with "Unknown"
df = df.fillna("Unknown", subset=categorical_cols)

print("Missing values filled successfully")

# Verify remaining missing values
remaining_missing = []

for c in df.columns:
    missing_count = df.filter(col(c).isNull()).count()
    if missing_count > 0:
        remaining_missing.append((c, missing_count))

print("Remaining missing values:", remaining_missing)

from pyspark.sql.functions import col

# Step 10: Feature Engineering

# Age in years
df = df.withColumn("AGE_YEARS", (-col("DAYS_BIRTH") / 365).cast("int"))

# Employment years
df = df.withColumn("EMPLOYMENT_YEARS", (-col("DAYS_EMPLOYED") / 365).cast("int"))

# Credit vs Income (very important)
df = df.withColumn(
    "CREDIT_INCOME_RATIO",
    col("AMT_CREDIT") / col("AMT_INCOME_TOTAL")
)

# Annuity vs Income (repayment pressure)
df = df.withColumn(
    "ANNUITY_INCOME_RATIO",
    col("AMT_ANNUITY") / col("AMT_INCOME_TOTAL")
)

# Show new columns
df.select(
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO"
).show(5)

# Step 11: SQL Analysis

df.createOrReplaceTempView("credit_data")

# 1. Overall default rate
spark.sql("""
SELECT 
    TARGET,
    COUNT(*) AS total_customers
FROM credit_data
GROUP BY TARGET
""").show()

# 2. Average income by risk
spark.sql("""
SELECT 
    TARGET,
    AVG(AMT_INCOME_TOTAL) AS avg_income
FROM credit_data
GROUP BY TARGET
""").show()

# 3. Risk by education
spark.sql("""
SELECT 
    NAME_EDUCATION_TYPE,
    AVG(TARGET) AS default_rate
FROM credit_data
GROUP BY NAME_EDUCATION_TYPE
ORDER BY default_rate DESC
""").show()

# 4. Risk by occupation
spark.sql("""
SELECT 
    OCCUPATION_TYPE,
    AVG(TARGET) AS default_rate
FROM credit_data
GROUP BY OCCUPATION_TYPE
ORDER BY default_rate DESC
""").show()

# 5. High risk customers (important)
spark.sql("""
SELECT *
FROM credit_data
WHERE CREDIT_INCOME_RATIO > 5
LIMIT 10
""").show()

from pyspark.sql.functions import when

# Step 12: Create business risk segments

df = df.withColumn(
    "INCOME_SEGMENT",
    when(col("AMT_INCOME_TOTAL") < 100000, "Low Income")
    .when((col("AMT_INCOME_TOTAL") >= 100000) & (col("AMT_INCOME_TOTAL") < 200000), "Medium Income")
    .otherwise("High Income")
)

df = df.withColumn(
    "CREDIT_RISK_BAND",
    when(col("CREDIT_INCOME_RATIO") > 6, "High Credit Burden")
    .when((col("CREDIT_INCOME_RATIO") >= 3) & (col("CREDIT_INCOME_RATIO") <= 6), "Medium Credit Burden")
    .otherwise("Low Credit Burden")
)

df = df.withColumn(
    "ANNUITY_RISK_BAND",
    when(col("ANNUITY_INCOME_RATIO") > 0.3, "High Repayment Burden")
    .when((col("ANNUITY_INCOME_RATIO") >= 0.15) & (col("ANNUITY_INCOME_RATIO") <= 0.3), "Medium Repayment Burden")
    .otherwise("Low Repayment Burden")
)

df.select(
    "SK_ID_CURR",
    "TARGET",
    "INCOME_SEGMENT",
    "CREDIT_RISK_BAND",
    "ANNUITY_RISK_BAND"
).show(10)

# Step 13: Risk segment analysis

df.createOrReplaceTempView("credit_data")

spark.sql("""
SELECT 
    INCOME_SEGMENT,
    COUNT(*) AS total_customers,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM credit_data
GROUP BY INCOME_SEGMENT
ORDER BY default_rate_percent DESC
""").show()

spark.sql("""
SELECT 
    CREDIT_RISK_BAND,
    COUNT(*) AS total_customers,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM credit_data
GROUP BY CREDIT_RISK_BAND
ORDER BY default_rate_percent DESC
""").show()

spark.sql("""
SELECT 
    ANNUITY_RISK_BAND,
    COUNT(*) AS total_customers,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM credit_data
GROUP BY ANNUITY_RISK_BAND
ORDER BY default_rate_percent DESC
""").show()

# Step 14: Save processed dashboard dataset

dashboard_cols = [
    "SK_ID_CURR",
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "INCOME_SEGMENT",
    "CREDIT_RISK_BAND",
    "ANNUITY_RISK_BAND",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
    "CODE_GENDER"
]

dashboard_df = df.select(dashboard_cols)

# Convert to pandas
pandas_df = dashboard_df.toPandas()

# Save as CSV
pandas_df.to_csv(
    "data/processed/credit_dashboard_data.csv",
    index=False
)

print("Dashboard-ready data saved successfully (via pandas).")

print("Dashboard-ready data saved successfully.")

spark.stop()