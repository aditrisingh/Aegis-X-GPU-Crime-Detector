import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from xgboost import XGBClassifier
from sklearn.utils import resample

# 📂 Feature paths
crime_dir = "Output_IR3/Crime"
noc_dir = "Output_IR3/No Crime"

# 📦 Load .npy features
def load_features(folder, label):
    features, labels = [], []
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            data = np.load(path)
            features.append(data)
            labels.append(label)
    return features, labels

# 🚶‍♀ Load features
X_crime, y_crime = load_features(crime_dir, 1)
X_noc, y_noc = load_features(noc_dir, 0)

# 🧊 Combine
X = np.array(X_crime + X_noc)
y = np.array(y_crime + y_noc)

# 🪓 Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ⚖️ Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🏗️ Ensemble using 5 XGBoost classifiers
n_ensemble = 5
ensemble_models = []

for i in range(n_ensemble):
    # Bootstrap sampling
    X_sample, y_sample = resample(X_train_scaled, y_train, replace=True, random_state=42+i)
    
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42+i
    )
    clf.fit(X_sample, y_sample)
    ensemble_models.append(clf)
    print(f"🔹 Trained XGB {i+1}/{n_ensemble}")

# 🧮 Ensemble predictions
probs = np.zeros((X_test_scaled.shape[0], 2))
for model in ensemble_models:
    probs += model.predict_proba(X_test_scaled)
probs /= n_ensemble
y_pred = np.argmax(probs, axis=1)

# 📊 Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"✅ Ensemble XGBoost Test Accuracy: {acc:.4f}")

# 💾 Save models and scaler
for idx, model in enumerate(ensemble_models):
    joblib.dump(model, f"xgb_ensemble_model_{idx+1}.joblib")
joblib.dump(scaler, "scaler_xgb.joblib")
print(f"✅ Ensemble XGBoost models and scaler saved successfully.")
