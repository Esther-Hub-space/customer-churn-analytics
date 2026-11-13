
-- name: customer_overview
SELECT
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned_customers,
    COUNT(*) - SUM(ChurnBinary) AS retained_customers,
    ROUND(
        100.0 * SUM(ChurnBinary)
        / NULLIF(COUNT(*), 0), 2
    ) AS churn_rate_pct,
    ROUND(
        100.0 * (COUNT(*) - SUM(ChurnBinary))
        / NULLIF(COUNT(*), 0), 2
    ) AS retained_share_pct
FROM customers;


-- name: churn_by_contract
SELECT
    Contract AS contract,
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned_customers,
    COUNT(*) - SUM(ChurnBinary) AS retained_customers,
    ROUND(
        100.0 * AVG(ChurnBinary), 2
    ) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- name: churn_by_tenure
WITH tenure_segments AS (
    SELECT
        CASE
            WHEN tenure <= 12 THEN '0-12 months'
            WHEN tenure <= 24 THEN '13-24 months'
            WHEN tenure <= 48 THEN '25-48 months'
            ELSE '49+ months'
        END AS tenure_group,
        CASE
            WHEN tenure <= 12 THEN 1
            WHEN tenure <= 24 THEN 2
            WHEN tenure <= 48 THEN 3
            ELSE 4
        END AS sort_order,
        ChurnBinary
    FROM customers
)
SELECT
    tenure_group,
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned_customers,
    ROUND(
        100.0 * AVG(ChurnBinary), 2
    ) AS churn_rate_pct
FROM tenure_segments
GROUP BY tenure_group, sort_order
ORDER BY sort_order;


-- name: churn_by_internet_service
SELECT
    InternetService AS internet_service,
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned_customers,
    ROUND(
        100.0 * AVG(ChurnBinary), 2
    ) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;


-- name: churn_by_payment_method
SELECT
    PaymentMethod AS payment_method,
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned_customers,
    ROUND(
        100.0 * AVG(ChurnBinary), 2
    ) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- name: churn_by_gender
SELECT
    gender,
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned_customers,
    ROUND(
        100.0 * AVG(ChurnBinary), 2
    ) AS churn_rate_pct
FROM customers
GROUP BY gender
ORDER BY churn_rate_pct DESC;


-- name: churn_by_senior_citizen
SELECT
    CASE
        WHEN SeniorCitizen = 1 THEN 'Senior'
        ELSE 'Non-senior'
    END AS customer_group,
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned_customers,
    ROUND(
        100.0 * AVG(ChurnBinary), 2
    ) AS churn_rate_pct
FROM customers
GROUP BY SeniorCitizen
ORDER BY churn_rate_pct DESC;


-- name: revenue_exposure
SELECT
    COUNT(*) AS total_customers,
    ROUND(
        SUM(MonthlyCharges), 2
    ) AS total_monthly_charges,
    ROUND(
        SUM(
            CASE
                WHEN ChurnBinary = 1
                THEN MonthlyCharges
                ELSE 0
            END
        ), 2
    ) AS churned_monthly_charges,
    ROUND(
        SUM(
            CASE
                WHEN ChurnBinary = 0
                THEN MonthlyCharges
                ELSE 0
            END
        ), 2
    ) AS retained_monthly_charges,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN ChurnBinary = 1
                THEN MonthlyCharges
                ELSE 0
            END
        ) / NULLIF(SUM(MonthlyCharges), 0),
        2
    ) AS exposure_share_pct
FROM customers;


-- name: revenue_exposure_by_contract
SELECT
    Contract AS contract,
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned_customers,
    ROUND(
        SUM(
            CASE
                WHEN ChurnBinary = 1
                THEN MonthlyCharges
                ELSE 0
            END
        ), 2
    ) AS churned_monthly_charges,
    ROUND(
        AVG(MonthlyCharges), 2
    ) AS average_monthly_charges
FROM customers
GROUP BY Contract
ORDER BY churned_monthly_charges DESC;


-- name: top_churned_customers_by_charges
SELECT
    customerID AS customer_id,
    Contract AS contract,
    tenure,
    MonthlyCharges AS monthly_charges,
    TotalCharges AS total_charges,
    InternetService AS internet_service
FROM customers
WHERE ChurnBinary = 1
ORDER BY MonthlyCharges DESC, customerID
LIMIT 20;


-- name: contract_payment_segments
SELECT
    Contract AS contract,
    PaymentMethod AS payment_method,
    COUNT(*) AS total_customers,
    SUM(ChurnBinary) AS churned_customers,
    ROUND(
        100.0 * AVG(ChurnBinary), 2
    ) AS churn_rate_pct
FROM customers
GROUP BY Contract, PaymentMethod
ORDER BY churn_rate_pct DESC;


-- name: data_quality_checks
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customerID) AS unique_customers,
    SUM(
        CASE WHEN TotalCharges IS NULL
        THEN 1 ELSE 0 END
    ) AS missing_total_charges,
    SUM(
        CASE
            WHEN ChurnBinary NOT IN (0, 1)
                 OR ChurnBinary IS NULL
            THEN 1 ELSE 0
        END
    ) AS invalid_churn_labels,
    SUM(
        CASE
            WHEN MonthlyCharges < 0
                 OR MonthlyCharges IS NULL
            THEN 1 ELSE 0
        END
    ) AS invalid_monthly_charges
FROM customers;
