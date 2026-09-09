
-- Query 1: Overall class balance of the target variable
SELECT
    stroke,
    COUNT(*) AS n_patients,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM clinical_records), 2) AS pct
FROM clinical_records
GROUP BY stroke;


-- Query 2: How much data is missing for each variable?
SELECT
    COUNT(*) AS total_records,
    SUM(CASE WHEN bmi IS NULL THEN 1 ELSE 0 END) AS missing_bmi,
    ROUND(100.0 * SUM(CASE WHEN bmi IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_missing_bmi
FROM clinical_records;

-- Query 3: Stroke rate by hypertension and heart disease status.
SELECT
    hypertension,
    heart_disease,
    COUNT(*) AS n_patients,
    ROUND(100.0 * AVG(stroke), 2) AS stroke_rate_pct
FROM clinical_records
GROUP BY hypertension, heart_disease
ORDER BY stroke_rate_pct DESC;

-- Query 4: Stroke rate by age bucket.
SELECT
    CASE
        WHEN p.age < 40 THEN '1. Under 40'
        WHEN p.age < 65 THEN '2. 40-64'
        ELSE '3. 65+'
    END AS age_group,
    COUNT(*) AS n_patients,
    ROUND(100.0 * AVG(c.stroke), 2) AS stroke_rate_pct
FROM patients p
JOIN clinical_records c ON p.patient_id = c.patient_id
GROUP BY age_group
ORDER BY age_group;

-- Query 5: Average glucose level and stroke rate by gender and work type.
SELECT
    p.gender,
    p.work_type,
    COUNT(*) AS n_patients,
    ROUND(AVG(c.avg_glucose_level), 1) AS avg_glucose,
    ROUND(100.0 * AVG(c.stroke), 2) AS stroke_rate_pct
FROM patients p
JOIN clinical_records c ON p.patient_id = c.patient_id
GROUP BY p.gender, p.work_type
ORDER BY stroke_rate_pct DESC;

-- Query 6: Which single age group has the highest stroke rate?
WITH age_group_stats AS (
    SELECT
        CASE
            WHEN p.age < 40 THEN '1. Under 40'
            WHEN p.age < 65 THEN '2. 40-64'
            ELSE '3. 65+'
        END AS age_group,
        COUNT(*) AS n_patients,
        AVG(c.stroke) AS stroke_rate
    FROM patients p
    JOIN clinical_records c ON p.patient_id = c.patient_id
    GROUP BY age_group
)
SELECT *
FROM age_group_stats
WHERE stroke_rate = (SELECT MAX(stroke_rate) FROM age_group_stats);