import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# -------------------------
# Load dataset
# -------------------------
def load_features(base_path):
    X, y = [], []
    for label, folder in enumerate(["No Crime", "Crime"]):
        folder_path = os.path.join(base_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".npy"):
                data = np.load(os.path.join(folder_path, file))
                X.append(data)
                y.append(label)
    return np.array(X), np.array(y)

base_path = "Output_IR3"  # Change if needed
X, y = load_features(base_path)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------
# Scale features
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# PCA (optional but helps high-dim data)
# -------------------------
pca = PCA(n_components=0.95, random_state=42)  # keep 95% variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# -------------------------
# Define models & parameters
# -------------------------
models_params = {
    "SVM": (SVC(), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }),
    "RandomForest": (RandomForestClassifier(random_state=42), {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20]
    }),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    })
}

# -------------------------
# Train & evaluate
# -------------------------
best_model = None
best_acc = 0
results = {}

for name, (model, params) in models_params.items():
    print(f"\nTraining {name}...")
    grid = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"Best Params: {grid.best_params_}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = grid.best_estimator_

# -------------------------
# Save the best model
# -------------------------
joblib.dump({"model": best_model, "scaler": scaler, "pca": pca}, "best_crimeeee.joblib")
print("\nBest Model Saved ✅")
print("All Results:", results)
