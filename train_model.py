import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("diabetic_data.csv")

# Feature Engineering
df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

def categorize_age(age):
    if age in ['[0-10)', '[10-20)', '[20-30)']:
        return 'Young'
    elif age in ['[30-40)', '[40-50)', '[50-60)']:
        return 'Middle-aged'
    else:
        return 'Elderly'

df['age_group'] = df['age'].apply(categorize_age)
df.drop('age', axis=1, inplace=True)

# Select Features
features = [
    'race','gender','time_in_hospital','num_lab_procedures',
    'num_procedures','num_medications',
    'number_outpatient','number_emergency','number_inpatient',
    'diag_1','diag_2','diag_3',
    'metformin','insulin','change','diabetesMed',
    'total_visits','age_group'
]

df = df[features + ['readmitted']]

df['readmitted'] = df['readmitted'].map({
    '<30':1,
    '>30':0,
    'NO':0
})

X = df.drop('readmitted', axis=1)
y = df['readmitted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64','float64']).columns

preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

smote = SMOTE(random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

pipeline.named_steps['classifier'].fit(X_train_res, y_train_res)

joblib.dump(pipeline, "model.pkl")

print(" Model saved as model.pkl successfully!")