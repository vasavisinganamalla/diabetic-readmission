from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Collect form data
    input_data = {
        'race': request.form['race'],
        'gender': request.form['gender'],
        'time_in_hospital': int(request.form['time_in_hospital']),
        'num_lab_procedures': int(request.form['num_lab_procedures']),
        'num_procedures': int(request.form['num_procedures']),
        'num_medications': int(request.form['num_medications']),
        'number_outpatient': int(request.form['number_outpatient']),
        'number_emergency': int(request.form['number_emergency']),
        'number_inpatient': int(request.form['number_inpatient']),
        'diag_1': request.form['diag_1'],
        'diag_2': request.form['diag_2'],
        'diag_3': request.form['diag_3'],
        'metformin': request.form['metformin'],
        'insulin': request.form['insulin'],
        'change': request.form['change'],
        'diabetesMed': request.form['diabetesMed'],
        'age_group': request.form['age_group']
    }

    # Create total_visits
    input_data['total_visits'] = (
        input_data['number_outpatient'] +
        input_data['number_emergency'] +
        input_data['number_inpatient']
    )

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]
    probability_percent = round(probability * 100, 2)

    # Risk categorization
    if probability_percent < 30:
        risk_level = "Very Low"
        confidence = "High"
    elif probability_percent < 50:
        risk_level = "Low"
        confidence = "Moderate"
    elif probability_percent < 70:
        risk_level = "Moderate"
        confidence = "Moderate"
    else:
        risk_level = "High"
        confidence = "High"

    return render_template(
        "index.html",
        prediction_text=f"""
        Prediction Result:<br>
        Risk Level: {risk_level}<br>
        Probability of Readmission: {probability_percent}%<br>
        Model Confidence: {confidence}
        """
    )

if __name__ == "__main__":
    app.run(debug=True)