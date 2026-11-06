import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier

def main():
    print("1. Loading dataset...")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    print("2. Cleaning data...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    print("3. Engineering features...")
    df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 48, 100], labels=['0-12 months', '13-48 months', '>48 months']).astype(str)
    df['monthly_charge_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 150], labels=['Low', 'Medium', 'High']).astype(str)
    df['has_dependents'] = (df['Dependents'] == 'Yes').astype(int)
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = (df[service_cols] == 'Yes').sum(axis=1)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    print("4. Preparing features & splitting data...")
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'total_services']
    
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    feature_names = X_encoded.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print("5. Training and comparing models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }
    for name, m in models.items():
        m.fit(X_train_scaled, y_train)
        preds = m.predict(X_test_scaled)
        probas = m.predict_proba(X_test_scaled)[:, 1]
        print(f"   {name:20s} | F1: {f1_score(y_test, preds):.4f} | ROC-AUC: {roc_auc_score(y_test, probas):.4f}")
        
    print("6. Hyperparameter tuning LightGBM with GridSearchCV...")
    param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.03, 0.05, 0.1], 'max_depth': [3, 5, 7]}
    grid = GridSearchCV(LGBMClassifier(random_state=42, verbose=-1), param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    final_model = grid.best_estimator_
    print(f"   Best Params: {grid.best_params_}")
    
    print("\n7. Final Model Classification Report:")
    print(classification_report(y_test, final_model.predict(X_test_scaled), target_names=['No Churn', 'Churn']))
    
    print("8. Saving model and preprocessing pipeline...")
    joblib.dump(final_model, 'best_churn_model.joblib')
    joblib.dump({
        'scaler': scaler,
        'feature_names': feature_names,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'service_cols': service_cols
    }, 'preprocessing_pipeline.joblib')
    print("✓ Project pipeline finished! Run `streamlit run app.py` to launch the web app.")

if __name__ == '__main__':
    main()
