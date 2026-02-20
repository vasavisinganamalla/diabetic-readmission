from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib

app = Flask(__name__)

model = joblib.load("gradient_boosting_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")
scaler = joblib.load("standard_scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        data = {
            'race': request.form['race'],
            'gender': request.form['gender'],
            'age_group': request.form['age_group'],
            'time_in_hospital': int(request.form['time_in_hospital']),
            'num_lab_procedures': int(request.form['num_lab_procedures']),
            'num_procedures': int(request.form['num_procedures']),
            'num_medications': int(request.form['num_medications']),
            'number_outpatient': int(request.form['number_outpatient']),
            'number_emergency': int(request.form['number_emergency']),
            'number_inpatient': int(request.form['number_inpatient']),
            'total_visits': int(request.form['total_visits']),
            'diag_1': request.form['diag_1'],
            'diag_2': request.form['diag_2'],
            'diag_3': request.form['diag_3'],
            'metformin': request.form['metformin'],
            'insulin': request.form['insulin'],
            'change': request.form['change'],
            'diabetesMed': request.form['diabetesMed']
        }

        input_df = pd.DataFrame([data])

        # Apply encoder
        categorical_cols = ['race', 'gender', 'age_group',
                            'diag_1', 'diag_2', 'diag_3',
                            'metformin', 'insulin', 'change', 'diabetesMed']

        input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])

        # Apply scaler
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        result = "Patient likely to be readmitted" if prediction == 1 else "Patient not likely to be readmitted"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)