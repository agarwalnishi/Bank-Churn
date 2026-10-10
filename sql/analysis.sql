-- =====================================================================
-- Bank Customer Churn: segment-level analysis (SQLite)
-- Table: customers  (built by src/etl.py)
-- Each query is preceded by a "-- name:" tag so run_analysis.py can
-- execute them one by one and save the results.
-- =====================================================================


-- name: 01_overall_churn
-- Baseline: how many customers left, and how much balance left with them.
SELECT
    COUNT(*)                                             AS customers,
    SUM(Exited)                                          AS churned,
    ROUND(100.0 * AVG(Exited), 2)                        AS churn_rate_pct,
    ROUND(SUM(CASE WHEN Exited = 1 THEN Balance END), 2) AS lost_deposits,
    ROUND(100.0 * SUM(CASE WHEN Exited = 1 THEN Balance END) / SUM(Balance), 2)
                                                         AS pct_of_total_deposits_lost
FROM customers;


-- name: 02_churn_by_geography
-- Churn rate and lost deposits per country, ranked by churn rate.
WITH geo AS (
    SELECT
        Geography,
        COUNT(*)                                             AS customers,
        SUM(Exited)                                          AS churned,
        ROUND(100.0 * AVG(Exited), 2)                        AS churn_rate_pct,
        ROUND(SUM(CASE WHEN Exited = 1 THEN Balance ELSE 0 END), 2) AS lost_deposits
    FROM customers
    GROUP BY Geography
)
SELECT
    *,
    RANK() OVER (ORDER BY churn_rate_pct DESC)               AS churn_rank,
    ROUND(100.0 * lost_deposits / SUM(lost_deposits) OVER (), 2) AS share_of_lost_deposits_pct
FROM geo
ORDER BY churn_rank;


-- name: 03_churn_by_activity
-- Active vs inactive members, with the inactive/active churn multiple.
WITH act AS (
    SELECT
        CASE WHEN IsActiveMember = 1 THEN 'Active' ELSE 'Inactive' END AS member_status,
        COUNT(*)                      AS customers,
        SUM(Exited)                   AS churned,
        AVG(Exited)                   AS churn_rate
    FROM customers
    GROUP BY IsActiveMember
)
SELECT
    member_status,
    customers,
    churned,
    ROUND(100.0 * churn_rate, 2) AS churn_rate_pct,
    ROUND(churn_rate / MIN(churn_rate) OVER (), 2) AS churn_multiple_vs_lowest
FROM act
ORDER BY churn_rate_pct DESC;


-- name: 04_churn_by_products
-- Churn by number of products held; 3+ products is the danger zone.
SELECT
    NumOfProducts,
    COUNT(*)                                  AS customers,
    SUM(Exited)                               AS churned,
    ROUND(100.0 * AVG(Exited), 2)             AS churn_rate_pct,
    CASE
        WHEN NumOfProducts >= 3 THEN 'High risk'
        WHEN NumOfProducts = 1  THEN 'Moderate risk'
        ELSE 'Low risk'
    END                                       AS risk_label
FROM customers
GROUP BY NumOfProducts
ORDER BY NumOfProducts;


-- name: 05_churn_by_age_group
-- Age is the strongest single driver; LAG shows the step-change between bands.
WITH ag AS (
    SELECT
        AgeGroup,
        COUNT(*)                      AS customers,
        ROUND(100.0 * AVG(Exited), 2) AS churn_rate_pct
    FROM customers
    GROUP BY AgeGroup
)
SELECT
    AgeGroup,
    customers,
    churn_rate_pct,
    ROUND(churn_rate_pct - LAG(churn_rate_pct) OVER (ORDER BY AgeGroup), 2)
                                  AS change_vs_prev_band_pp
FROM ag
ORDER BY AgeGroup;


-- name: 06_balance_quartiles
-- NTILE(4) over customers who hold a balance: does churn rise with balance?
WITH q AS (
    SELECT
        Balance,
        Exited,
        NTILE(4) OVER (ORDER BY Balance) AS balance_quartile
    FROM customers
    WHERE Balance > 0
)
SELECT
    balance_quartile,
    COUNT(*)                         AS customers,
    ROUND(MIN(Balance), 0)           AS min_balance,
    ROUND(MAX(Balance), 0)           AS max_balance,
    ROUND(100.0 * AVG(Exited), 2)    AS churn_rate_pct
FROM q
GROUP BY balance_quartile
UNION ALL
SELECT
    0, COUNT(*), 0, 0, ROUND(100.0 * AVG(Exited), 2)
FROM customers
WHERE Balance = 0
ORDER BY balance_quartile;


-- name: 07_credit_score_deciles
-- NTILE(10) on credit score: credit quality is a weak churn signal.
WITH d AS (
    SELECT CreditScore, Exited,
           NTILE(10) OVER (ORDER BY CreditScore) AS credit_decile
    FROM customers
)
SELECT
    credit_decile,
    MIN(CreditScore)                AS min_score,
    MAX(CreditScore)                AS max_score,
    COUNT(*)                        AS customers,
    ROUND(100.0 * AVG(Exited), 2)   AS churn_rate_pct
FROM d
GROUP BY credit_decile
ORDER BY credit_decile;


-- name: 08_geography_x_gender
-- Rank gender segments within each country (PARTITION BY).
WITH seg AS (
    SELECT
        Geography,
        Gender,
        COUNT(*)                      AS customers,
        ROUND(100.0 * AVG(Exited), 2) AS churn_rate_pct
    FROM customers
    GROUP BY Geography, Gender
)
SELECT
    *,
    RANK() OVER (PARTITION BY Geography ORDER BY churn_rate_pct DESC) AS rank_in_country,
    ROUND(churn_rate_pct - AVG(churn_rate_pct) OVER (PARTITION BY Geography), 2)
                                  AS diff_vs_country_avg_pp
FROM seg
ORDER BY Geography, rank_in_country;


-- name: 09_top_balance_customers_per_country
-- Top 5 balances per country that already churned: who should retention have called?
WITH ranked AS (
    SELECT
        CustomerId,
        Geography,
        Age,
        NumOfProducts,
        IsActiveMember,
        ROUND(Balance, 2) AS Balance,
        ROW_NUMBER() OVER (PARTITION BY Geography ORDER BY Balance DESC) AS rn
    FROM customers
    WHERE Exited = 1
)
SELECT * FROM ranked
WHERE rn <= 5
ORDER BY Geography, rn;


-- name: 10_tenure_churn_running
-- Churn by tenure year with a running cumulative share of all churners.
WITH t AS (
    SELECT
        Tenure,
        COUNT(*)                      AS customers,
        SUM(Exited)                   AS churned,
        ROUND(100.0 * AVG(Exited), 2) AS churn_rate_pct
    FROM customers
    GROUP BY Tenure
)
SELECT
    Tenure,
    customers,
    churned,
    churn_rate_pct,
    SUM(churned) OVER (ORDER BY Tenure ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                                  AS cumulative_churned,
    ROUND(100.0 * SUM(churned) OVER (ORDER BY Tenure) / SUM(churned) OVER (), 2)
                                  AS cumulative_share_pct
FROM t
ORDER BY Tenure;


-- name: 11_risk_segment_matrix
-- Rule-based risk segments built with CASE, then sized by customers and deposits at risk.
WITH flagged AS (
    SELECT
        *,
        CASE
            WHEN NumOfProducts >= 3                              THEN '1. 3+ products'
            WHEN Age >= 45 AND IsActiveMember = 0                THEN '2. 45+ and inactive'
            WHEN Geography = 'Germany' AND IsActiveMember = 0    THEN '3. Germany and inactive'
            WHEN IsActiveMember = 0                              THEN '4. Other inactive'
            ELSE                                                      '5. Active, low risk'
        END AS risk_segment
    FROM customers
)
SELECT
    risk_segment,
    COUNT(*)                                                  AS customers,
    SUM(Exited)                                               AS churned,
    ROUND(100.0 * AVG(Exited), 2)                             AS churn_rate_pct,
    ROUND(SUM(CASE WHEN Exited = 1 THEN Balance ELSE 0 END), 0) AS lost_deposits,
    ROUND(SUM(CASE WHEN Exited = 0 THEN Balance ELSE 0 END), 0) AS deposits_still_at_risk
FROM flagged
GROUP BY risk_segment
ORDER BY risk_segment;


-- name: 12_churner_vs_retained_profile
-- Side-by-side profile of churned vs retained customers.
SELECT
    CASE WHEN Exited = 1 THEN 'Churned' ELSE 'Retained' END AS status,
    COUNT(*)                                  AS customers,
    ROUND(AVG(Age), 1)                        AS avg_age,
    ROUND(AVG(CreditScore), 0)                AS avg_credit_score,
    ROUND(AVG(Balance), 0)                    AS avg_balance,
    ROUND(AVG(EstimatedSalary), 0)            AS avg_salary,
    ROUND(AVG(NumOfProducts), 2)              AS avg_products,
    ROUND(100.0 * AVG(IsActiveMember), 1)     AS pct_active,
    ROUND(100.0 * AVG(ZeroBalance), 1)        AS pct_zero_balance,
    ROUND(100.0 * AVG(GenderMale), 1)         AS pct_male
FROM customers
GROUP BY Exited
ORDER BY status;
