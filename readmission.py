import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

#loading Dataset
df = pd.read_csv("diabetic_data.csv")
# Quick overview of dataset structure
df.head(10).T
df.shape
df.nunique()
#features in particular column
# Check distribution of target variable
print("\nReadmission Counts:")
display(df['readmitted'].value_counts())
print("\nReadmission Percentage:")
print(df['readmitted'].value_counts(normalize=True)*100)
#visualization of target variable
plt.figure(figsize=(5,4))
sns.countplot(x='readmitted', data=df)
plt.title("Readmission Count (0 = No, 1 = Yes)")
plt.show()
#correlation matrix
# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=np.number)

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Create the heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap (Numeric Columns Only)")
plt.show()
## Age vs Readmission Distribution
plt.figure(figsize=(15,4))
sns.histplot(data=df, x='age', hue='readmitted', kde=True, multiple='stack')
plt.title("Age Distribution by Readmission")
plt.show()

#data preprocessing
# Find columns where every value is unique
unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]

print("Unwanted (unique) columns:", unique_cols)
#constant or repeated same values in row
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print("Unwanted (constant) columns:", constant_cols)
# Find columns with more than 40% missing values
missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0.4]
print("Columns with too many missing values:", missing_cols)
# Drop them
df = df.drop(unique_cols + constant_cols + missing_cols, axis=1)
# Safely combine and drop
to_drop = list(set(unique_cols + constant_cols + missing_cols))
df.drop(columns=to_drop, inplace=True, errors='ignore')

print(f"Dropped {len(to_drop)} columns")
print("New dataset shape:", df.shape)
print("Remaining columns:", len(df.columns))
print("Remaining rows:", len(df))
print("Total missing values after cleanup:")
print(df.isnull().sum())
#sum of duplicates
df.T.duplicated().sum()
#Handle Missing Values Automatically
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns
# Fill numeric columns with median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode (most frequent value)
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("missing values filled")
print("\nFinal shape after preprocessing:", df.shape)
# List of columns you want to keep
selected_cols = [
    'race', 'gender', 'age', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'diag_1', 'diag_2', 'diag_3','number_diagnoses',
    'metformin', 'insulin',
    'change', 'diabetesMed', 'readmitted'
]

print(df.columns)

# Find columns to drop (everything not in selected list)
df = df[selected_cols].copy()
print("Selected Features Shape:", df.shape)
# Convert target variable to binary
# 1 → readmitted within 30 days, 0 → not readmitted or >30
df['readmitted'] = df['readmitted'].map({
    '<30': 1,
    '>30': 0,
    'NO': 0
})

print("Target distribution AFTER conversion:")
print(df['readmitted'].value_counts())

sns.countplot(x='readmitted', data=df)
plt.title("Final Target Distribution")
plt.show()
# feature engineering.............................
# Combine Visit Counts
df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

# Categorize Age
def categorize_age(age):
    if age in ['[0-10)', '[10-20)', '[20-30)']:
        return 'Young'
    elif age in ['[30-40)', '[40-50)', '[50-60)']:
        return 'Middle-aged'
    elif age in ['[60-70)', '[70-80)', '[80-90)', '[90-100)']:
        return 'Elderly'
    else:
        return 'Unknown'

df['age_group'] = df['age'].apply(categorize_age)

# Drop the original 'age' column
df = df.drop('age', axis=1)
print("Feature engineering completed")

# Choose relevant features (make sure these columns exist in your df)
features = ['race', 'gender', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'diag_1', 'diag_2', 'diag_3',
    'metformin', 'insulin',
    'change', 'diabetesMed', 'readmitted',
    'total_visits', 'age_group' 
    # Include the new engineered features
]

#  Keep only these features
df = df[features].copy()

# Handle missing values safely 
print("Missing values before modeling:")
print(df.isnull().sum())
print("Data filtered and target column converted successfully!")
print("Final shape:", df.shape)
print(df.head())
print("\n Data After Preprocessing:")
print(df.info())

sns.countplot(x='readmitted', data=df)
plt.title("Readmission Distribution After Cleaning")
plt.show()
#Prepare Data for Modeling
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Encode categorical variables
# Train-Test Split FIRST
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Now encode ONLY on training data
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

categorical_cols = X_train.select_dtypes(include='object').columns
numerical_cols = X_train.select_dtypes(include=['int64','float64']).columns

encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

# Encode categorical columns
X_train.loc[:, categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
X_test.loc[:, categorical_cols] = encoder.transform(X_test[categorical_cols])

# Scale numerical columns
scaler = StandardScaler()

X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])

print("y_train distribution BEFORE SMOTE:")
print(y_train.value_counts())
#smote 
#Handle Class Imbalance using SMOTE
# Train-Test Split FIRST (Very Important)
   # keeps class distribution balanced
# Apply SMOTE ONLY on training data
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())

# SMOTE Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x=y, hue=y, palette='Set2')
plt.title("Before SMOTE")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_res, hue=y_train_res, palette='Set1')
plt.title("After SMOTE (Training Data Only)")
plt.show()

print(y_train_res.value_counts())

# Model Training
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train_res, y_train_res)
print("Model trained.")
# Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
y_pred = model.predict(X_test)
print("\n Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
precision = precision_score(y_test, y_pred, zero_division=1)
print("precesion=",precision)
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Save Model & Preprocessing Objects
import joblib
joblib.dump(model, 'gradient_boosting_model.pkl')
joblib.dump(encoder, 'ordinal_encoder.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')
print(" Model and preprocessing tools saved.")