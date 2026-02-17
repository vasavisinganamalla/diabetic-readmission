import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display

#loading Dataset
df = pd.read_csv("diabetic_data.csv")
df.head(10).T
df.shape
df.nunique()
#features in particular column
display(df['readmitted'].value_counts())
#visualization of target variable
plt.figure(figsize=(5,4))
sns.countplot(x='readmitted', data=df)
plt.title("Readmission Count (0 = No, 1 = Yes)")
plt.show()

print(df['readmitted'].value_counts(normalize=True)*100)
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
#histogram of age distribution
plt.figure(figsize=(15,4))
sns.histplot(data=df, x='age', hue='readmitted', kde=True, multiple='stack')
plt.title("Age Distribution by Readmission")
plt.show()
#sum of duplicates
df.T.duplicated().sum()
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
#Handle Missing Values Automatically
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Fill numeric columns with median
for col in num_cols:
    df[col].fillna(df[col].median())

# Fill categorical columns with mode (most frequent value)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0])
print("missing values filled")
#Encode Categorical Columns (Convert text → numbers)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Identify categorical columns in the current DataFrame
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
print("Categorical columns encoded.")
#Scale numeric features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("Numeric features scaled.")

print("\nFinal shape after preprocessing:", df.shape)
#to check the actual features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Separate features (X) and target (y)
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Identify categorical columns in X
categorical_cols = X.select_dtypes(include='object').columns

# Apply Label Encoding to categorical columns in X
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# Encode the target variable 'readmitted'
y_encoded = le.fit_transform(y.astype(str))


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded) # Use y_encoded for training

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 20 Important Features:\n")
print(importances.head(20))

plt.figure(figsize=(8,5))
sns.barplot(x=importances.head(20), y=importances.head(20).index)
plt.title("Top 20 Important Features")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
# List of columns you want to keep
selected_cols = [
    'race', 'gender', 'age', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'diag_1', 'diag_2', 'diag_3','number_diagnoses'
    'max_glu_serum', 'A1Cresult',
    'metformin', 'insulin',
    'change', 'diabetesMed', 'readmitted'
]

# Find columns to drop (everything not in selected list)
cols_to_drop = [col for col in df.columns if col not in selected_cols]

# Drop them
df = df.drop(cols_to_drop, axis=1)

# Confirm
print("Dropped columns:", cols_to_drop)
print("Total dropped:", len(cols_to_drop))
print("Remaining columns:", len(df.columns))
print("Remaining rows:", df.shape[0])
#checking shape of the dataset after cleaning
df.shape
# Calculate counts
num_removed = len(cols_to_drop)
num_remaining = len(df.columns)

# Data for the bar chart
labels = ['Removed Columns', 'Remaining Columns']
values = [num_removed, num_remaining]

# Create bar plot
plt.figure(figsize=(15,8))
plt.bar(labels, values, color=['skyblue', 'grey'])

# Add titles and labels
plt.title('Comparison of Removed vs Remaining Columns', fontsize=14)
plt.ylabel('Number of Columns', fontsize=12)
plt.xlabel('Column Category', fontsize=12)

# Add value labels on bars
for i, v in enumerate(values):
    plt.text(i, v + 0.3, str(v), ha='center', fontsize=12)

# Display the plot
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
# Step 1: Choose relevant features (make sure these columns exist in your df)
features = ['race', 'gender', 'time_in_hospital',
    'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient',
    'diag_1', 'diag_2', 'diag_3',
    'metformin', 'insulin',
    'change', 'diabetesMed', 'readmitted',
    'total_visits', 'age_group' # Include the new engineered features
]



# Step 2: Keep only these features
df = df[features].copy()

# Step 3: Convert target variable to binary
# 1 → readmitted within 30 days, 0 → not readmitted or >30
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Step 4: Handle missing values safely (drop rows with any NaN)
df.dropna(inplace=True)

print("Data filtered and target column converted successfully!")
print("Final shape:", df.shape)
print(df.head())
print("\n Data After Preprocessing:")
print(df.info())

sns.countplot(x='readmitted', data=df)
plt.title("Readmission Distribution After Cleaning")
plt.show()
X = df.drop('readmitted', axis=1)
y = df['readmitted']

categorical = X.select_dtypes(include='object').columns.tolist()
numerical = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical] = encoder.fit_transform(X[categorical])
# Reload the original dataframe to get the original 'readmitted' column
df_original = pd.read_csv("disease_data.csv")

# Extract the original 'readmitted' column
y_original = df_original['readmitted']

print("Original 'readmitted' column loaded.")
display(y_original.head())
#smote 
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X[categorical], y_original)

# SMOTE Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x=y_original,hue=y_original,palette='Set2')
plt.title("Before SMOTE")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=y_res,hue=y_res,palette='Set1')
plt.title("After SMOTE")
plt.show()
y_res = y_res.apply(lambda x: 1 if x == '<30' else 0)
display(y_res.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.4, random_state=42)
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model trained.")
import joblib
joblib.dump(model, 'gradient_boosting_model.pkl')
joblib.dump(encoder, 'ordinal_encoder.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')
print(" Model and preprocessing tools saved.")
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
from sklearn.model_selection import train_test_split

print("\n Accuracy Across Multiple Random Splits:")
accuracies = []

for seed in range(10):  # Try 10 different splits
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=seed)
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Seed {seed}: Accuracy = {acc:.4f}")

print(f"\n Average Accuracy over 10 runs: {np.mean(accuracies):.4f}")
