SELECT 
  TARGET,
  COUNT(*) AS total_customers
FROM workspace.default.application_train
GROUP BY TARGET;


SELECT 
  OCCUPATION_TYPE,
  COUNT(*) AS total_customers,
  ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM workspace.default.application_train
GROUP BY OCCUPATION_TYPE
ORDER BY default_rate_percent DESC
LIMIT 15;

SELECT 
  OCCUPATION_TYPE,
  COUNT(*) AS total_customers,
  ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM workspace.default.application_train
GROUP BY OCCUPATION_TYPE
ORDER BY default_rate_percent DESC
LIMIT 15;

SELECT 
  NAME_INCOME_TYPE,
  COUNT(*) AS total_customers,
  ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM workspace.default.application_train
GROUP BY NAME_INCOME_TYPE
ORDER BY default_rate_percent DESC;

SELECT 
  CODE_GENDER,
  COUNT(*) AS total_customers,
  ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM workspace.default.application_train
GROUP BY CODE_GENDER
ORDER BY default_rate_percent DESC;


SELECT 
  CASE 
    WHEN AMT_CREDIT < 300000 THEN 'Low Credit'
    WHEN AMT_CREDIT BETWEEN 300000 AND 800000 THEN 'Medium Credit'
    ELSE 'High Credit'
  END AS credit_amount_band,
  COUNT(*) AS total_customers,
  ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM workspace.default.application_train
GROUP BY credit_amount_band
ORDER BY default_rate_percent DESC;


SELECT 
  CASE 
    WHEN AMT_CREDIT < 300000 THEN 'Low Credit'
    WHEN AMT_CREDIT BETWEEN 300000 AND 800000 THEN 'Medium Credit'
    ELSE 'High Credit'
  END AS credit_amount_band,
  COUNT(*) AS total_customers,
  ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM workspace.default.application_train
GROUP BY credit_amount_band
ORDER BY default_rate_percent DESC;

SELECT 
  CASE 
    WHEN AMT_CREDIT / AMT_INCOME_TOTAL > 6 THEN 'High Credit Burden'
    WHEN AMT_CREDIT / AMT_INCOME_TOTAL BETWEEN 3 AND 6 THEN 'Medium Credit Burden'
    ELSE 'Low Credit Burden'
  END AS credit_income_band,
  COUNT(*) AS total_customers,
  ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM workspace.default.application_train
WHERE AMT_INCOME_TOTAL > 0
GROUP BY credit_income_band
ORDER BY default_rate_percent DESC;

SELECT 
  CASE 
    WHEN AMT_ANNUITY / AMT_INCOME_TOTAL > 0.3 THEN 'High Repayment Burden'
    WHEN AMT_ANNUITY / AMT_INCOME_TOTAL BETWEEN 0.15 AND 0.3 THEN 'Medium Repayment Burden'
    ELSE 'Low Repayment Burden'
  END AS repayment_burden_band,
  COUNT(*) AS total_customers,
  ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM workspace.default.application_train
WHERE AMT_INCOME_TOTAL > 0
GROUP BY repayment_burden_band
ORDER BY default_rate_percent DESC;

SELECT 
  CASE 
    WHEN -DAYS_BIRTH / 365 < 30 THEN 'Below 30'
    WHEN -DAYS_BIRTH / 365 BETWEEN 30 AND 45 THEN '30-45'
    WHEN -DAYS_BIRTH / 365 BETWEEN 46 AND 60 THEN '46-60'
    ELSE 'Above 60'
  END AS age_group,
  COUNT(*) AS total_customers,
  ROUND(AVG(TARGET) * 100, 2) AS default_rate_percent
FROM workspace.default.application_train
GROUP BY age_group
ORDER BY default_rate_percent DESC;