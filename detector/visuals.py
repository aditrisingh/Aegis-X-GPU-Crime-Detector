import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, classification_report, average_precision_score
import joblib
import os

# === Load Test Data ===
print("[INFO] Loading test data...")
X_test, y_test = [], []
labels_map = {"No Crime": 0, "Crime": 1}

test_dir = "Test Features"
for label_name, label_value in labels_map.items():
    folder_path = os.path.join(test_dir, label_name)
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            feat = np.load(os.path.join(folder_path, file))
            X_test.append(feat)
            y_test.append(label_value)

X_test = np.array(X_test)
y_test = np.array(y_test)

# === Load Model + Scaler + PCA ===
print("[INFO] Loading model, scaler, PCA from joblib...")
bundle = joblib.load("best_crime_detector2.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
pca = bundle["pca"]

# === Preprocess ===
X_scaled = scaler.transform(X_test)
X_pca = pca.transform(X_scaled)

# === Predictions ===
print("[INFO] Running predictions...")
y_pred = model.predict(X_pca)

# Some models may not have predict_proba
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_pca)[:, 1]
else:
    from sklearn.preprocessing import MinMaxScaler
    scores = model.decision_function(X_pca)
    y_prob = MinMaxScaler().fit_transform(scores.reshape(-1, 1)).ravel()

# === Print Report ===
report = classification_report(y_test, y_pred, target_names=["No Crime", "Crime"])
print("\nClassification Report:\n", report)

# === Metrics for Plots ===
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap_score = average_precision_score(y_test, y_prob)

# === God-Tier Performance Report ===
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Confusion Matrix (Counts + %)
sns.heatmap(cm, annot=False, cmap="Blues", cbar=False, ax=axes[0,0])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0,0].text(j+0.5, i+0.5, f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)",
                       ha="center", va="center", color="black", fontsize=12)
axes[0,0].set_title("Confusion Matrix", fontsize=16)
axes[0,0].set_xlabel("Predicted Label")
axes[0,0].set_ylabel("True Label")
axes[0,0].set_xticklabels(["No Crime", "Crime"])
axes[0,0].set_yticklabels(["No Crime", "Crime"], rotation=0)

# 2. ROC Curve
axes[0,1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
axes[0,1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
axes[0,1].set_title("ROC Curve", fontsize=16)
axes[0,1].set_xlabel("False Positive Rate")
axes[0,1].set_ylabel("True Positive Rate")
axes[0,1].legend(loc="lower right")

# 3. Precision–Recall Curve
axes[1,0].plot(recall, precision, color="purple", lw=2, label=f"AP = {ap_score:.2f}")
axes[1,0].set_title("Precision–Recall Curve", fontsize=16)
axes[1,0].set_xlabel("Recall")
axes[1,0].set_ylabel("Precision")
axes[1,0].legend(loc="lower left")

# 4. PCA Explained Variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
axes[1,1].bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, color="teal", label="Individual")
axes[1,1].plot(range(1, len(explained_variance)+1), cumulative_variance, color="red", marker="o", label="Cumulative")
axes[1,1].set_title("PCA Explained Variance", fontsize=16)
axes[1,1].set_xlabel("Principal Components")
axes[1,1].set_ylabel("Variance Ratio")
axes[1,1].legend()

plt.tight_layout()
plt.savefig("god_tier_performance_report.png", dpi=300)
plt.show()

print("[INFO] God-tier visual saved as god_tier_performance_report.png")
