"""
Test that model.pkl and the Flask app at http://127.0.0.1:5000/ predict correctly.
Run from project folder: python test_predictions.py
- With server running: also tests /predict API.
- Without server: validates model on test set only.
"""
import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# Run from project directory so model.pkl and diabetic_data.csv are found
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# Same preprocessing as train_model.py
def load_and_prepare_data():
    df = pd.read_csv("diabetic_data.csv")
    df['total_visits'] = (
        df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    )
    def categorize_age(age):
        if age in ['[0-10)', '[10-20)', '[20-30)']:
            return 'Young'
        elif age in ['[30-40)', '[40-50)', '[50-60)']:
            return 'Middle-aged'
        else:
            return 'Elderly'
    df['age_group'] = df['age'].apply(categorize_age)
    df.drop('age', axis=1, inplace=True)
    features = [
        'race', 'gender', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'diag_1', 'diag_2', 'diag_3',
        'metformin', 'insulin', 'change', 'diabetesMed',
        'total_visits', 'age_group'
    ]
    df = df[features + ['readmitted']]
    df['readmitted'] = df['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})
    df = df.dropna(subset=['readmitted'])
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    return X, y

def main():
    print("Loading model and data...")
    try:
        pipeline = joblib.load("model.pkl")
    except Exception as e:
        print(f"Could not load model.pkl: {e}")
        print("Tip: Ensure pandas/sklearn versions match the environment that saved the model.")
        _test_api_only()
        print("\nDone.")
        return
    X, y = load_and_prepare_data()

    # Same split as train_model.py (test_size=0.2, random_state=42)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- Model validation on held-out test set ---")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # P(Readmit <30)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=['No readmit', 'Readmit <30']))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Spot-check: a few rows prediction vs actual
    print("\n--- Spot check (first 5 test rows) ---")
    for i in range(min(5, len(X_test))):
        row = X_test.iloc[i:i+1]
        actual = y_test.iloc[i]
        pred = pipeline.predict(row)[0]
        prob = pipeline.predict_proba(row)[0][1]
        ok = "OK" if pred == actual else "MISMATCH"
        print(f"  Row {i}: actual={actual}, predicted={pred}, prob(readmit)={prob:.3f} -> {ok}")

    _test_api_only(pipeline=pipeline, X_test=X_test)
    print("\nDone.")

def _test_api_only(pipeline=None, X_test=None):
    """Hit /predict and optionally compare with local pipeline."""
    from sklearn.model_selection import train_test_split
    try:
        import requests
    except ImportError:
        print("\n--- API test skipped (install requests) ---")
        return
    # One row for API: from test set if available, else from data
    X, y = load_and_prepare_data()
    _, X_test_use, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sample = X_test_use.iloc[0]
    age_map = {'Young': '20-30', 'Middle-aged': '40-50', 'Elderly': '70-80'}
    payload = {
        'race': str(sample['race']),
        'gender': str(sample['gender']),
        'time_in_hospital': int(sample['time_in_hospital']),
        'num_lab_procedures': int(sample['num_lab_procedures']),
        'num_procedures': int(sample['num_procedures']),
        'num_medications': int(sample['num_medications']),
        'number_outpatient': int(sample['number_outpatient']),
        'number_emergency': int(sample['number_emergency']),
        'number_inpatient': int(sample['number_inpatient']),
        'diag_1': str(sample['diag_1']),
        'diag_2': str(sample['diag_2']),
        'diag_3': str(sample['diag_3']),
        'metformin': str(sample['metformin']),
        'insulin': str(sample['insulin']),
        'change': str(sample['change']),
        'diabetesMed': str(sample['diabetesMed']),
        'age_group': age_map.get(str(sample['age_group']), '40-50'),
    }
    try:
        r = requests.post("http://127.0.0.1:5000/predict", data=payload, timeout=5)
        r.raise_for_status()
        print("\n--- API test (http://127.0.0.1:5000/predict) ---")
        if pipeline is not None and X_test is not None:
            expected_prob = pipeline.predict_proba(X_test.iloc[0:1])[0][1]
            print(f"  Local pipeline P(readmit): {expected_prob:.4f}")
        print(f"  Response status: {r.status_code}")
        if "Probability of Readmission:" in r.text:
            print("  API returned a prediction (probability shown on page).")
        else:
            print("  Response snippet:", r.text[:200])
    except requests.exceptions.ConnectionError:
        print("\n--- API test skipped (Flask server not running at http://127.0.0.1:5000) ---")
    except Exception as e:
        print(f"\n--- API test error: {e} ---")

if __name__ == "__main__":
    main()
