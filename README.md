# patient-readmission-project
predicting readmissions of patients using ML and SMOTE

Machine Learning model to predict hospital readmission risk for diabetic patients using structured clinical data.

Overview

This project builds a binary classification model that predicts whether a diabetic patient will be readmitted within 30 days of discharge.

Early prediction of readmission risk helps improve patient care and reduce hospital costs.

Problem Type

Binary Classification

0 → No Readmission

1 → Readmission

Tech Stack

Python

Pandas

NumPy

Scikit-learn

Gradient Boosting Classifier

Project Files

diabetic_data.csv – Dataset

gradient_boosting_model.pkl – Trained model

ordinal_encoder.pkl – Categorical encoder

standard_scaler.pkl – Feature scaler

readmission.py – Prediction script

How to Run

Clone the repository:

git clone https://github.com/vasavisinganamalla/diabetic-readmission.git
cd diabetic-readmission

Install dependencies:

pip install pandas numpy scikit-learn

Run the model:

python readmission.py
Model

Gradient Boosting Classifier trained on patient demographic, clinical, and hospital visit features.

Use Case

Identify high-risk patients

Support hospital decision-making

Improve healthcare outcomes