import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("       LOAN APPROVAL PREDICTION - ML PIPELINE")
print("=" * 60)

# ── 1. DATA COLLECTION ────────────────────────────────────────────
print("\n[1] DATA COLLECTION")
print("-" * 40)
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())

# ── 2. DATA PREPROCESSING ─────────────────────────────────────────
print("\n[2] DATA PREPROCESSING")
print("-" * 40)

# Check missing values before
print("Missing values BEFORE cleaning:")
print(df.isnull().sum())

# Fill missing values
df['Gender']           = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']          = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']       = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']    = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount']       = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History']   = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

# Encode categorical columns
le = LabelEncoder()
for col in ['Gender', 'Married', 'Dependents', 'Education', 
            'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = le.fit_transform(df[col].astype(str))

print("\nDataset after encoding:")
print(df.head())

# Feature engineering


print("\nNew features added:")
print("  - Total_Income = ApplicantIncome + CoapplicantIncome")
print("  - Income_Loan_Ratio = Total_Income / LoanAmount")
print("  - EMI = LoanAmount / Loan_Amount_Term")

# Features and target
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

print(f"\nFeatures used: {list(X.columns)}")
print(f"Target distribution:\n{y.value_counts()}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} samples")
print(f"Test size:  {X_test.shape[0]} samples")

# ── 3. MODEL TRAINING ─────────────────────────────────────────────
print("\n[3] MODEL TRAINING")
print("-" * 40)

# Model 1 — Random Forest
print("Training Model 1: Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
print("Random Forest trained!")

# Model 2 — Logistic Regression
print("Training Model 2: Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000, 
    random_state=42,
    class_weight='balanced'
)
lr_model.fit(X_train, y_train)
print("Logistic Regression trained!")

# ── 4. MODEL EVALUATION ───────────────────────────────────────────
print("\n[4] MODEL EVALUATION")
print("-" * 40)

def evaluate_model(name, model, X_test, y_test, X, y):
    preds = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy:           {accuracy_score(y_test, preds)*100:.2f}%")
    print(f"  Precision:          {precision_score(y_test, preds)*100:.2f}%")
    print(f"  Recall:             {recall_score(y_test, preds)*100:.2f}%")
    print(f"  F1 Score:           {f1_score(y_test, preds)*100:.2f}%")
    print(f"  Cross-val (5-fold): {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(f"  True Negative  (Correctly Rejected): {cm[0][0]}")
    print(f"  False Positive (Wrongly Approved):   {cm[0][1]}")
    print(f"  False Negative (Wrongly Rejected):   {cm[1][0]}")
    print(f"  True Positive  (Correctly Approved): {cm[1][1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds, 
          target_names=['Rejected', 'Approved']))
    
    return accuracy_score(y_test, preds)

rf_acc = evaluate_model("Random Forest",        rf_model, X_test, y_test, X, y)
lr_acc = evaluate_model("Logistic Regression",  lr_model, X_test, y_test, X, y)

# ── 5. RESULT ANALYSIS ────────────────────────────────────────────
print("\n[5] RESULT ANALYSIS")
print("-" * 40)
print(f"\n  Random Forest Accuracy:       {rf_acc*100:.2f}%")
print(f"  Logistic Regression Accuracy: {lr_acc*100:.2f}%")

if rf_acc >= lr_acc:
    best_model = rf_model
    best_name  = "Random Forest"
    best_acc   = rf_acc
else:
    best_model = lr_model
    best_name  = "Logistic Regression"
    best_acc   = lr_acc

print(f"\n  WINNER: {best_name} with {best_acc*100:.2f}% accuracy!")

# Feature importance (Random Forest only)
print(f"\n  Feature Importance (Random Forest):")
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<25} {bar} {imp:.4f}")

# ── 6. SAVE MODELS ────────────────────────────────────────────
print("\n[6] SAVING MODELS")
print("-" * 40)

# ✅ Save BOTH models (IMPORTANT)
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(lr_model, "lr_model.pkl")

print("  Random Forest saved as rf_model.pkl")
print("  Logistic Regression saved as lr_model.pkl")

# ✅ Save feature names separately (VERY IMPORTANT)
feature_names = list(X.columns)
joblib.dump(feature_names, "feature_names.pkl")

print("  Feature names saved as feature_names.pkl")