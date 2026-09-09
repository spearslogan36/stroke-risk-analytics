DROP TABLE IF EXISTS clinical_records;
DROP TABLE IF EXISTS patients;


-- Creates the table for storing patient information
CREATE TABLE patients (
    patient_id        INTEGER PRIMARY KEY,
    gender            TEXT NOT NULL,
    age               REAL NOT NULL,
    ever_married      TEXT NOT NULL,
    work_type         TEXT NOT NULL,
    residence_type    TEXT NOT NULL
);

-- Creates the table for storing clinical records of patients
CREATE TABLE clinical_records (
    record_id         INTEGER PRIMARY KEY,
    patient_id        INTEGER NOT NULL,
    hypertension      INTEGER NOT NULL,
    heart_disease     INTEGER NOT NULL,
    avg_glucose_level REAL NOT NULL,
    bmi               REAL,
    smoking_status    TEXT NOT NULL,
    stroke            INTEGER NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX idx_clinical_patient_id ON clinical_records(patient_id);