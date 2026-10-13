"""Single source of truth for the feature schema (labels, categories, units, widget types).
Add a clinical feature by adding an entry here (and a column in the data) - the API and UI adapt."""

LEAKAGE_COLUMNS = ["LAD", "LCX", "RCA", "Cath"]  # targets; never used as inputs
TARGETS = {"cad": "Cath", "lad": "LAD", "lcx": "LCX", "rca": "RCA"}
POSITIVE = {"Cath": "CAD", "LAD": "Stenotic", "LCX": "Stenotic", "RCA": "Stenotic"}

# name: (label, category, unit, kind, options)  kind: number | binary | ordinal
F = {}
def add(name, label, cat, unit="", kind="number", options=None):
    F[name] = dict(label=label, category=cat, unit=unit, kind=kind, options=options)

YN = {0: "No", 1: "Yes"}
add("Age", "Age", "Demographic", "years")
add("Sex", "Sex", "Demographic", kind="binary", options={0: "Female", 1: "Male"})
add("Weight", "Weight", "Demographic", "kg")
add("Length", "Height", "Demographic", "cm")
add("BMI", "BMI", "Demographic", "kg/m²")
for n, l in [("DM", "Diabetes mellitus"), ("HTN", "Hypertension"), ("Current Smoker", "Current smoker"),
             ("EX-Smoker", "Ex-smoker"), ("FH", "Family history of CAD"), ("Obesity", "Obesity"),
             ("CRF", "Chronic renal failure"), ("CVA", "Stroke history (CVA)"),
             ("Airway disease", "Airway disease"), ("Thyroid Disease", "Thyroid disease"),
             ("CHF", "Congestive heart failure"), ("DLP", "Dyslipidemia")]:
    add(n, l, "History & risk factors", kind="binary", options=YN)
add("BP", "Systolic blood pressure", "Clinical exam", "mmHg")
add("PR", "Pulse rate", "Clinical exam", "bpm")
for n, l in [("Edema", "Edema"), ("Weak Peripheral Pulse", "Weak peripheral pulse"), ("Lung rales", "Lung rales"),
             ("Systolic Murmur", "Systolic murmur"), ("Diastolic Murmur", "Diastolic murmur"),
             ("Typical Chest Pain", "Typical chest pain"), ("Dyspnea", "Dyspnea"),
             ("Atypical", "Atypical chest pain"), ("Nonanginal", "Non-anginal chest pain"),
             ("LowTH Ang", "Low-threshold angina")]:
    add(n, l, "Clinical exam", kind="binary", options=YN)
add("Function Class", "Functional class (0-3)", "Clinical exam", kind="ordinal",
    options={0: "0", 1: "I", 2: "II", 3: "III"})
for n, l in [("Q Wave", "Q wave"), ("St Elevation", "ST elevation"), ("St Depression", "ST depression"),
             ("Tinversion", "T-wave inversion"), ("LVH", "Left ventricular hypertrophy"),
             ("Poor R Progression", "Poor R-wave progression"), ("LBBB", "Left bundle branch block"),
             ("RBBB", "Right bundle branch block")]:
    add(n, l, "ECG", kind="binary", options=YN)
for n, l, u in [("FBS", "Fasting blood sugar", "mg/dL"), ("CR", "Creatinine", "mg/dL"), ("TG", "Triglycerides", "mg/dL"),
                ("LDL", "LDL cholesterol", "mg/dL"), ("HDL", "HDL cholesterol", "mg/dL"), ("BUN", "Blood urea nitrogen", "mg/dL"),
                ("ESR", "ESR", "mm/h"), ("HB", "Hemoglobin", "g/dL"), ("K", "Potassium", "mEq/L"), ("Na", "Sodium", "mEq/L"),
                ("WBC", "White blood cells", "cells/µL"), ("Lymph", "Lymphocytes", "%"), ("Neut", "Neutrophils", "%"),
                ("PLT", "Platelets", "10³/µL")]:
    add(n, l, "Laboratory", u)
add("EF-TTE", "Ejection fraction (echo)", "Echocardiography", "%")
add("Region RWMA", "Regions with wall-motion abnormality", "Echocardiography", "regions (0-4)", kind="ordinal",
    options={0: "0", 1: "1", 2: "2", 3: "3", 4: "4"})
add("VHD", "Valvular heart disease", "Echocardiography", kind="ordinal",
    options={0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"})
