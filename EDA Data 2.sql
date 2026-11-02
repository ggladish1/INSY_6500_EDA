WITH study_patients AS (
    SELECT
        patient_id,
        DENSE_RANK() OVER (ORDER BY patient_id) AS study_id
    FROM patients
),
historical_labs AS (
    SELECT
        lp.patient_id,
        lp.source_date AS lab_date,

        MAX(CASE
            WHEN lr.lab_test_id = 19
            THEN lr.value_numeric
        END) AS creatinine,

        MAX(CASE
            WHEN lr.lab_test_id = 23
            THEN lr.value_numeric
        END) AS egfr,

        MAX(CASE
            WHEN lr.lab_test_id = 1
            THEN lr.value_numeric
        END) AS acr,

        MAX(CASE
            WHEN lr.lab_test_id = 25
            THEN lr.value_numeric
        END) AS glucose,

        MAX(CASE
            WHEN lr.lab_test_id = 15
            THEN lr.value_numeric
        END) AS potassium,

        MAX(CASE
            WHEN lr.lab_test_id = 6
            THEN lr.value_numeric
        END) AS hemoglobin,

        MAX(CASE
            WHEN lr.lab_test_id = 20
            THEN lr.value_numeric
        END) AS calcium,

        MAX(CASE
            WHEN lr.lab_test_id = 21
            THEN lr.value_numeric
        END) AS phosphorus,

		MAX(CASE
            WHEN lr.lab_test_id = 17
            THEN lr.value_numeric
        END) AS co2,

		MAX(CASE
            WHEN lr.lab_test_id = 22
            THEN lr.value_numeric
        END) AS albumin

    FROM lab_panels lp

    JOIN lab_results lr
        ON lr.lab_panel_id = lp.lab_panel_id

    WHERE lr.lab_test_id IN (
        1, 6, 15, 17, 27, 19, 20, 21, 22, 23, 25
    )
      AND lp.source_date IS NOT NULL

    GROUP BY
        lp.patient_id,
        lp.source_date
),

/* ---------------------------------------------------------
   Collapse vitals to one row per encounter.
   --------------------------------------------------------- */
encounter_vitals AS (
    SELECT
        encounter_id,
        MAX(bp_sys) AS systolic_bp,
        MAX(bp_dia) AS diastolic_bp,
        MAX(bmi) AS bmi,
        MAX(weight_lbs) AS weight_lbs

    FROM vital_sets

    WHERE encounter_id IS NOT NULL

    GROUP BY encounter_id
)

SELECT
    sp.study_id,

    /* Demographics */
    p.sex,
    p.race,
    p.ethnicity,
	p.date_of_birth,

    /* Historical observation date */
    hl.lab_date,

    /* Historical labs */
    hl.creatinine,
    hl.egfr,
    hl.acr,
    hl.glucose,
    hl.potassium,
    hl.hemoglobin,
    hl.calcium,
    hl.phosphorus,
	hl.co2,
	hl.albumin,

    /* Information from closest preceding encounter */
    enc.encounter_date,
    enc.location,
    enc.insurance,

    enc.systolic_bp,
    enc.diastolic_bp,
    enc.bmi,
    enc.weight_lbs

FROM historical_labs hl

JOIN patients p
    ON p.patient_id = hl.patient_id

JOIN study_patients sp
    ON sp.patient_id = hl.patient_id

/* ---------------------------------------------------------
   Find the most recent encounter occurring on or before
   this laboratory date.

   LATERAL allows this lookup to happen independently for
   every historical laboratory observation.
   --------------------------------------------------------- */
LEFT JOIN LATERAL (

    SELECT
        e.encounter_date,
        e.location_raw AS location,
        e.insurance_company AS insurance,

        ev.systolic_bp,
        ev.diastolic_bp,
        ev.bmi,
        ev.weight_lbs

    FROM encounters e

    LEFT JOIN encounter_vitals ev
        ON ev.encounter_id = e.encounter_id

    WHERE e.patient_id = hl.patient_id
      AND e.encounter_date IS NOT NULL
      AND e.encounter_date <= hl.lab_date

    ORDER BY e.encounter_date DESC

    LIMIT 1

) enc ON TRUE

ORDER BY
    sp.study_id,
    hl.lab_date;