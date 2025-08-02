# credit_scoring_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)
import joblib

# 1. Load or Create Dataset
data = pd.DataFrame({
    "income": [50000, 60000, 40000, 80000, 30000],
    "debt_ratio": [0.3, 0.4, 0.5, 0.2, 0.6],
    "monthly_expense": [1500, 1600, 1400, 2000, 1200],
    "payment_history_score": [0.9, 0.85, 0.7, 0.95, 0.65],
    "credit_card_usage": [0.4, 0.6, 0.7, 0.3, 0.8],
    "num_defaults": [0, 1, 2, 0, 3],
    "creditworthy": [1, 1, 0, 1, 0]
})

# 2. Split Features and Target
X = data.drop("creditworthy", axis=1)
y = data["creditworthy"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Classifier (Random Forest Recommended)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("🔍 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("🎯 ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# 7. Convert Probabilities to Credit Score (300–850 scale)
def to_credit_score(prob, min_score=300, max_score=850):
    return int(min_score + (max_score - min_score) * prob)

for i, prob in enumerate(y_proba):
    score = to_credit_score(prob)
    print(f"Person {i+1} → Probability: {prob:.2f} → Credit Score: {score}")

# 8. Save Model and Scaler
joblib.dump(model, "credit_score_model.pkl")
joblib.dump(scaler, "credit_scaler.pkl")

print("\n✅ Model and Scaler saved!")
