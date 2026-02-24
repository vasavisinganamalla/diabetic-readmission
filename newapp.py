from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

FEATURE_ORDER = [
    'race', 'gender', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'diag_1', 'diag_2', 'diag_3',
    'metformin', 'insulin', 'change', 'diabetesMed',
    'total_visits', 'age_group'
]

def _form_age_to_group(age_str):
    if age_str in ('0-10', '10-20', '20-30'):
        return 'Young'
    if age_str in ('30-40', '40-50', '50-60'):
        return 'Middle-aged'
    return 'Elderly'


# ✅ HOME PAGE
@app.route("/")
def home():
    return render_template("home.html")


# ✅ FORM PAGE
@app.route("/form")
def form():
    return render_template("form.html")


# ✅ PREDICTION
@app.route("/predict", methods=["POST"])
def predict():

    number_outpatient = int(request.form['number_outpatient'])
    number_emergency = int(request.form['number_emergency'])
    number_inpatient = int(request.form['number_inpatient'])
    total_visits = number_outpatient + number_emergency + number_inpatient

    input_data = {
        'race': request.form['race'],
        'gender': request.form['gender'],
        'time_in_hospital': int(request.form['time_in_hospital']),
        'num_lab_procedures': int(request.form['num_lab_procedures']),
        'num_procedures': int(request.form['num_procedures']),
        'num_medications': int(request.form['num_medications']),
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'diag_1': request.form['diag_1'],
        'diag_2': request.form['diag_2'],
        'diag_3': request.form['diag_3'],
        'metformin': request.form['metformin'],
        'insulin': request.form['insulin'],
        'change': request.form['change'],
        'diabetesMed': request.form['diabetesMed'],
        'total_visits': total_visits,
        'age_group': _form_age_to_group(request.form['age_group'])
    }

    input_df = pd.DataFrame([input_data])[FEATURE_ORDER]

    probability = model.predict_proba(input_df)[0][1]
    probability_percent = round(probability * 100, 2)

    if probability_percent < 30:
        risk_level = "Very Low"
    elif probability_percent < 50:
        risk_level = "Low"
    elif probability_percent < 70:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return render_template(
        "result.html",
        risk_level=risk_level,
        probability=probability_percent
    )


# 🚀 ALWAYS LAST
if __name__ == "__main__":
    app.run(debug=True)