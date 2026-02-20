import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

# Load data
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/baseline_model.pkl")

# Save baseline stats
X_train.describe().to_csv("models/baseline_stats.csv")

print("✅ Baseline model trained and saved.")