import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# ==============================
# 1️⃣ LOAD DATA
# ==============================
data = pd.read_csv("data/loan_data_set.csv")

# fill missing values
data = data.ffill()

# ==============================
# 2️⃣ ENCODE CATEGORICAL DATA
# ==============================
categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

le = LabelEncoder()

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# remove ID column
data.drop("Loan_ID", axis=1, inplace=True)

# ==============================
# 3️⃣ SPLIT FEATURES & TARGET
# ==============================
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4️⃣ TRAIN BASELINE MODEL
# ==============================
model = XGBClassifier()
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)

print("Baseline AUC:", auc)

# ==============================
# 5️⃣ STORE BASELINE STATS
# ==============================
baseline_stats = X_train.describe()

# ==============================
# 6️⃣ SIMULATE NEW DATA (DRIFT)
# ==============================
new_data = X_test.copy()

# create artificial drift
new_data["ApplicantIncome"] = new_data["ApplicantIncome"] * 1.5

# ==============================
# 7️⃣ DRIFT DETECTION
# ==============================
old_mean = X_train["ApplicantIncome"].mean()
new_mean = new_data["ApplicantIncome"].mean()

change_percent = abs(new_mean - old_mean) / old_mean

print("Old Mean:", old_mean)
print("New Mean:", new_mean)
print("Change Percentage:", change_percent)

# ==============================
# 8️⃣ SEVERITY + ACTION
# ==============================
if change_percent < 0.1:
    severity = "🟢 Mild Drift"
    action = "No retraining needed"

elif change_percent < 0.3:
    severity = "🟡 Moderate Drift"
    action = "Sliding window retraining"

elif change_percent < 0.5:
    severity = "🟠 Severe Drift"
    action = "Feature evolution + Retraining"

else:
    severity = "🔴 Extreme Drift"
    action = "Full model retraining"

print(severity)
print("Action:", action)

# ==============================
# 9️⃣ AUTO RETRAIN (IF NEEDED)
# ==============================
if change_percent >= 0.3:

    print("🔄 Retraining model...")

    model.fit(new_data, y_test)

    new_preds = model.predict_proba(new_data)[:, 1]
    new_auc = roc_auc_score(y_test, new_preds)

    print("New Model AUC:", new_auc)

else:
    print("No retraining triggered.")
