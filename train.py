import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def main():
    print("1.=== Data Loading ===")
    df = pd.read_csv("data/Telco_Customer_Churn.csv")
    df = df.drop(columns=['customerID'])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df, drop_first=True)


    print("2.=== Splitting Feature & Targets ===")
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=42, stratify=y
    )

    print("3.=== Scaling Features ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("4.=== Model Training with LogisticRegression===")
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    print("5.=== Evaluation Model ===")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]

    print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC score : {roc_auc_score(y_test, y_proba):.4f}\n")
    print("==== Classification Report ===")
    print(classification_report(y_test, y_pred))


    print("6.=== Saving Model & Scaler ===")

    os.makedirs("model", exist_ok=True)
    print("6. === Saving Model & Scaler ===")
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model aur Scaler successfully save ho gaye!")



if __name__ == "__main__":
    main()




