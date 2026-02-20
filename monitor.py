import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def run_monitor():

    # Load model
    model = joblib.load("models/baseline_model.pkl")
    baseline_stats = pd.read_csv("models/baseline_stats.csv", index_col=0)

    # Load new data
    data = pd.read_csv("data/loan_data_set.csv")
    data = data.ffill()

    categorical_cols = [
        "Gender","Married","Dependents",
        "Education","Self_Employed",
        "Property_Area","Loan_Status"
    ]

    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    data.drop("Loan_ID", axis=1, inplace=True)

    X = data.drop("Loan_Status", axis=1)
    y = data["Loan_Status"]

    X_new = X.sample(frac=0.2, random_state=42)
    y_new = y.loc[X_new.index]

    # Simulate drift
    X_new["ApplicantIncome"] *= 5

    # Multi-feature drift
    drift_scores = []

    for col in X_new.columns:
        if col in baseline_stats.columns:
            old_mean = baseline_stats.loc["mean", col]
            new_mean = X_new[col].mean()

            if old_mean != 0:
                score = abs(new_mean - old_mean) / abs(old_mean)
                drift_scores.append(score)

    global_drift = sum(drift_scores) / len(drift_scores)

    # Severity
    if global_drift < 0.1:
        severity = "🟢 Mild Drift"
        action = "No retraining required"

    elif global_drift < 0.3:
        severity = "🟡 Moderate Drift"
        action = "Monitoring"

    elif global_drift < 0.5:
        severity = "🟠 Severe Drift"
        action = "Retraining triggered"

    else:
        severity = "🔴 Extreme Drift"
        action = "Full retraining"

    # Auto retrain if severe
    new_auc = None

    if global_drift >= 0.3:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        new_model = XGBClassifier()
        new_model.fit(X_train, y_train)

        preds = new_model.predict_proba(X_test)[:, 1]
        new_auc = roc_auc_score(y_test, preds)

        joblib.dump(new_model, "models/baseline_model.pkl")
        X_train.describe().to_csv("models/baseline_stats.csv")

    return global_drift, severity, action, new_auc